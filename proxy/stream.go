package proxy

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"syscall"

	"github.com/bornholm/genai/llm"
)

// streamEmitter encodes a stream of llm.StreamChunk values into a
// wire-format-specific sequence of SSE events.
//
// EmitFirst is called once, for the first chunk received from the
// provider (already validated to carry no error). Emit is called for
// each subsequent chunk. EmitError is called when a mid-stream chunk
// carries an error; no further calls are made afterwards. Finalize is
// called once after the loop ends, on the success path only, to write
// any closing events (e.g. "[DONE]" or "message_stop").
type streamEmitter interface {
	EmitFirst(w io.Writer, chunk llm.StreamChunk) error
	Emit(w io.Writer, chunk llm.StreamChunk) error
	EmitError(w io.Writer, err error) error
	Finalize(w io.Writer, usage llm.ChatCompletionUsage) error
}

// streamChatCompletion runs a streaming chat completion and encodes each
// chunk via emitter, writing SSE output to w. It handles the common
// concerns shared by all wire formats: peeking at the first chunk to
// return a proper HTTP error status if the backend rejects the request
// immediately, committing to SSE headers, flushing, and running
// post-response hooks with the accumulated usage.
func (s *Server) streamChatCompletion(
	w http.ResponseWriter,
	r *http.Request,
	req *ProxyRequest,
	client llm.ChatCompletionStreamingClient,
	resolvedModel string,
	opts []llm.ChatCompletionOptionFunc,
	emitter streamEmitter,
) {
	ctx := r.Context()

	chunks, err := client.ChatCompletionStream(ctx, opts...)
	if err != nil {
		slog.ErrorContext(ctx, "stream chat completion error", slog.Any("error", err))
		errRes, _ := s.chain.RunOnError(ctx, req, err)
		if errRes != nil {
			writeProxyResponse(w, errRes)
		} else {
			writeAPIError(w, apiErrorFromErr(err))
		}
		return
	}

	// Peek at the first chunk before committing to SSE headers.
	// This lets us return a proper HTTP error status when the backend
	// immediately rejects the request (e.g. invalid model parameters).
	firstChunk, ok := <-chunks
	if !ok {
		writeAPIError(w, NewInternalError("stream closed with no data"))
		return
	}
	if firstChunk.Error() != nil {
		slog.ErrorContext(ctx, "stream chunk error", slog.Any("error", firstChunk.Error()))
		errRes, _ := s.chain.RunOnError(ctx, req, firstChunk.Error())
		if errRes != nil {
			writeProxyResponse(w, errRes)
		} else {
			writeAPIError(w, apiErrorFromErr(firstChunk.Error()))
		}
		return
	}

	// First chunk is good — commit to SSE.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, canFlush := w.(http.Flusher)
	flush := func() {
		if canFlush {
			flusher.Flush()
		}
	}

	tracker := llm.NewStreamingUsageTracker()
	// interruption stays nil while the stream is on its normal course. It is
	// what tells the post-response hooks below that the usage they receive is
	// partial, and why.
	var interruption *StreamInterruption
	chunksEmitted := 0

	tracker.Update(firstChunk)
	if err := emitter.EmitFirst(w, firstChunk); err != nil {
		logStreamWriteError(ctx, "could not emit first stream chunk", err)
		// Once a write to the response fails the SSE stream is unrecoverable:
		// stop emitting instead of burning the rest of the upstream stream on a
		// connection nobody reads.
		interruption = &StreamInterruption{Cause: StreamInterruptionClientGone, Err: err}
	} else {
		chunksEmitted++
	}
	flush()

	if interruption == nil && !firstChunk.IsComplete() {
		for chunk := range chunks {
			if chunk.Error() != nil {
				slog.ErrorContext(ctx, "stream chunk error", slog.Any("error", chunk.Error()))
				// Headers already sent; forward the error as an event.
				if err := emitter.EmitError(w, chunk.Error()); err != nil {
					logStreamWriteError(ctx, "could not emit stream error", err)
				}
				flush()
				interruption = &StreamInterruption{Cause: StreamInterruptionUpstream, Err: chunk.Error()}
				break
			}

			tracker.Update(chunk)
			if err := emitter.Emit(w, chunk); err != nil {
				logStreamWriteError(ctx, "could not emit stream chunk", err)
				interruption = &StreamInterruption{Cause: StreamInterruptionClientGone, Err: err}
				break
			}
			chunksEmitted++
			flush()

			if chunk.IsComplete() {
				break
			}
		}
	}

	// An interrupted stream still delivered whatever the provider produced
	// before it stopped, and the provider billed it, so the usage collected so
	// far is reported to the hooks below either way. Only the closing events are
	// skipped: after a client hangup writing them would fail again, and after an
	// upstream error the client has already been sent an error event.
	if interruption == nil {
		if err := emitter.Finalize(w, tracker.Usage()); err != nil {
			logStreamWriteError(ctx, "could not finalize stream", err)
		}
		flush()
	} else {
		interruption.ChunksEmitted = chunksEmitted
	}

	usage := tracker.Usage()
	streamTokensUsed := &TokenUsage{
		PromptTokens:     int(usage.PromptTokens()),
		CompletionTokens: int(usage.CompletionTokens()),
		TotalTokens:      int(usage.TotalTokens()),
	}
	type cachedUsageStream interface{ CachedTokens() int64 }
	if cu, ok := usage.(cachedUsageStream); ok {
		streamTokensUsed.CachedTokens = int(cu.CachedTokens())
	}
	if cr, ok := usage.(llm.CostReportingUsage); ok {
		if amount, currency, ok := cr.Cost(); ok {
			streamTokensUsed.Cost = &amount
			streamTokensUsed.CostCurrency = currency
		}
	}
	// The status stays 200: the client was sent 200 headers the moment the first
	// chunk arrived, and an interruption cannot take them back. Interruption is
	// what tells a hook the exchange did not complete.
	proxyRes := &ProxyResponse{
		StatusCode:   http.StatusOK,
		Body:         nil,
		TokensUsed:   streamTokensUsed,
		Interruption: interruption,
	}
	if err := s.chain.RunPostResponse(ctx, req, proxyRes); err != nil {
		slog.WarnContext(ctx, "post-response hook error", slog.Any("error", err))
	}
}

// logStreamWriteError reports a failed write to the SSE response. A client
// hanging up mid-stream — a closed tab, an aborted request, a reverse proxy
// timing out — is ordinary traffic rather than a server fault, so it is logged
// at debug level and without the stack trace.
func logStreamWriteError(ctx context.Context, msg string, err error) {
	if isClientGone(ctx, err) {
		slog.DebugContext(ctx, msg+": client went away", slog.String("error", err.Error()))
		return
	}
	slog.ErrorContext(ctx, msg, slog.Any("error", err))
}

// isClientGone reports whether err means the HTTP client is no longer there to
// read the response.
func isClientGone(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if ctx.Err() != nil {
		return true
	}
	return errors.Is(err, syscall.EPIPE) ||
		errors.Is(err, syscall.ECONNRESET) ||
		errors.Is(err, net.ErrClosed) ||
		errors.Is(err, context.Canceled)
}

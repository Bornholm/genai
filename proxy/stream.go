package proxy

import (
	"io"
	"log/slog"
	"net/http"

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
	hadStreamError := false

	tracker.Update(firstChunk)
	if err := emitter.EmitFirst(w, firstChunk); err != nil {
		slog.ErrorContext(ctx, "could not emit first stream chunk", slog.Any("error", err))
	}
	flush()

	if !firstChunk.IsComplete() {
		for chunk := range chunks {
			if chunk.Error() != nil {
				slog.ErrorContext(ctx, "stream chunk error", slog.Any("error", chunk.Error()))
				// Headers already sent; forward the error as an event.
				if err := emitter.EmitError(w, chunk.Error()); err != nil {
					slog.ErrorContext(ctx, "could not emit stream error", slog.Any("error", err))
				}
				flush()
				hadStreamError = true
				break
			}

			tracker.Update(chunk)
			if err := emitter.Emit(w, chunk); err != nil {
				slog.ErrorContext(ctx, "could not emit stream chunk", slog.Any("error", err))
				continue
			}
			flush()

			if chunk.IsComplete() {
				break
			}
		}
	}

	// Skip finalization and post-response hooks on stream error — nothing useful to record.
	if hadStreamError {
		return
	}

	if err := emitter.Finalize(w, tracker.Usage()); err != nil {
		slog.ErrorContext(ctx, "could not finalize stream", slog.Any("error", err))
	}
	flush()

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
	proxyRes := &ProxyResponse{
		StatusCode: http.StatusOK,
		Body:       nil,
		TokensUsed: streamTokensUsed,
	}
	if err := s.chain.RunPostResponse(ctx, req, proxyRes); err != nil {
		slog.WarnContext(ctx, "post-response hook error", slog.Any("error", err))
	}
}

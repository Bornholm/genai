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
// called once after the loop ends, to write any closing events (e.g.
// "[DONE]" or "message_stop"), unless an error event was sent in their
// place or the client is no longer there to read them.
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

	// streamCtx is what the provider goroutine watches. Cancelling it on an
	// interruption is how the upstream stream is told to stop: a provider that
	// keeps producing into a channel nobody reads holds its HTTP response open
	// and goes on being billed.
	streamCtx, cancelStream := context.WithCancel(ctx)
	defer cancelStream()

	chunks, err := client.ChatCompletionStream(streamCtx, opts...)
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

	// abandonUpstream stops the provider and frees its goroutine. Cancelling
	// alone is not enough: a provider whose send does not watch its context
	// stays blocked on a full channel, so the remaining chunks are drained until
	// it closes the channel and its deferred cleanup runs.
	abandonUpstream := func() {
		cancelStream()
		go func() {
			// Bounded: a provider that honours its context closes the channel
			// right away, but a third-party client that watches neither its
			// context nor its consumer would otherwise keep this goroutine for
			// the whole life of the process. Giving up leaks what the previous
			// behaviour leaked anyway, and only for that kind of client.
			if !llm.DrainStream(chunks, s.options.DrainTimeout) {
				slog.WarnContext(ctx, "gave up draining an abandoned upstream stream",
					slog.Duration("after", s.options.DrainTimeout))
			}
		}()
	}

	tracker := llm.NewStreamingUsageTracker()
	// interruption stays nil while the stream is on its normal course. It is
	// what tells the post-response hooks below that the usage they receive is
	// partial, and why.
	var interruption *StreamInterruption
	chunksEmitted := 0
	// sawTerminal records that the provider signalled an end of its own, either
	// a completion chunk or an error. A stream that closes without one is
	// truncated, not complete.
	sawTerminal := false

	tracker.Update(firstChunk)
	if err := emitter.EmitFirst(w, firstChunk); err != nil {
		logStreamWriteError(ctx, "could not emit first stream chunk", err)
		// Once a write to the response fails the SSE stream is unrecoverable:
		// stop emitting instead of burning the rest of the upstream stream on a
		// connection nobody reads.
		interruption = &StreamInterruption{
			Cause: StreamInterruptionClientGone, Err: err, TerminalEventUndelivered: true,
		}
		abandonUpstream()
	} else {
		chunksEmitted++
	}
	flush()

	if firstChunk.IsComplete() {
		sawTerminal = true
	}

	if interruption == nil && !firstChunk.IsComplete() {
		for chunk := range chunks {
			if chunk.Error() != nil {
				sawTerminal = true
				slog.ErrorContext(ctx, "stream chunk error", slog.Any("error", chunk.Error()))
				// A provider that knows its counters attaches them to the error
				// chunk itself, so record it before anything else.
				tracker.Update(chunk)
				// Headers already sent; forward the error as an event.
				undelivered := false
				if err := emitter.EmitError(w, chunk.Error()); err != nil {
					logStreamWriteError(ctx, "could not emit stream error", err)
					// The client is gone too. The upstream failure is still what
					// stopped the stream, but a hook must not assume the error
					// event reached anyone.
					undelivered = true
				}
				flush()
				interruption = &StreamInterruption{
					Cause:                    StreamInterruptionUpstream,
					Err:                      chunk.Error(),
					TerminalEventUndelivered: undelivered,
				}
				// A conforming provider has closed its channel by now, so this
				// is a no-op for them; it bounds the damage for one that keeps
				// producing after its error chunk.
				abandonUpstream()
				break
			}

			tracker.Update(chunk)
			if err := emitter.Emit(w, chunk); err != nil {
				logStreamWriteError(ctx, "could not emit stream chunk", err)
				interruption = &StreamInterruption{
					Cause: StreamInterruptionClientGone, Err: err, TerminalEventUndelivered: true,
				}
				abandonUpstream()
				break
			}
			chunksEmitted++
			flush()

			if chunk.IsComplete() {
				sawTerminal = true
				break
			}
		}
	}

	// The channel closed without the provider ever signalling completion or
	// failure. The client keeps the normal closing events — a provider is free
	// to end a legitimate response without a terminal chunk, and sending an
	// error instead would let a client retry a response it received in full —
	// but the hooks are told, because a truncated stream is also what a dropped
	// upstream connection looks like and its usage is unknown.
	if interruption == nil && !sawTerminal {
		err := errors.New("upstream stream ended before completion")
		slog.WarnContext(ctx, "stream ended without a terminal chunk", slog.Any("error", err))
		interruption = &StreamInterruption{Cause: StreamInterruptionTruncated, Err: err}
	}

	// An interrupted stream still delivered whatever the provider produced
	// before it stopped, and the provider billed it, so the usage collected so
	// far is reported to the hooks below either way. The closing events are
	// skipped on the two paths where they cannot be written or would contradict
	// what the client was just sent: after a client hangup writing them would
	// fail again, and after an upstream error the client has already been sent
	// an error event.
	if interruption == nil || interruption.Cause == StreamInterruptionTruncated {
		if err := emitter.Finalize(w, tracker.Usage()); err != nil {
			logStreamWriteError(ctx, "could not finalize stream", err)
			if interruption == nil {
				// Everything was delivered but the terminator: the exchange did
				// not end normally for the client either.
				interruption = &StreamInterruption{Cause: StreamInterruptionClientGone, Err: err}
			}
			// On the truncated path the cause is already set; either way the
			// client did not get the closing events.
			interruption.TerminalEventUndelivered = true
		}
		flush()
	}
	if interruption != nil {
		interruption.ChunksEmitted = chunksEmitted
		interruption.PartialUsage = tracker.Reported()
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
	// The hooks run on a context detached from the request. On the client_gone
	// path r.Context() is already canceled — that is what ended the stream — and
	// a usage hook writing to a network or SQL store would fail on
	// context.Canceled, losing the very accounting this path exists to keep.
	// The budget replaces the request's own cancellation: without one, a hook
	// blocking on a dead store would hold this handler goroutine forever, which
	// ordinary client traffic could then exhaust.
	hookCtx := context.WithoutCancel(ctx)
	if s.options.PostResponseTimeout > 0 {
		var cancelHooks context.CancelFunc
		hookCtx, cancelHooks = context.WithTimeout(hookCtx, s.options.PostResponseTimeout)
		defer cancelHooks()
	}
	if err := s.chain.RunPostResponse(hookCtx, req, proxyRes); err != nil {
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

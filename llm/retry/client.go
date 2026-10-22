package retry

import (
	"context"
	stderrors "errors"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// DefaultDrainTimeout caps how long an abandoned attempt is drained for. A
// client that honours its context closes its channel at once; this only bounds
// what one that honours nothing can hold.
const DefaultDrainTimeout = 30 * time.Second

type Client struct {
	baseDelay    time.Duration
	maxRetries   int
	drainTimeout time.Duration
	client       llm.Client
}

// Options holds what NewClient does not take positionally.
type Options struct {
	// DrainTimeout caps how long the stream of an abandoned attempt is drained
	// for. Zero or less means no limit. Default DefaultDrainTimeout.
	DrainTimeout time.Duration
}

// OptionFunc is a functional option for the retrying client.
type OptionFunc func(*Options)

// WithDrainTimeout sets how long the stream of an abandoned attempt is drained
// for. Zero or less means no limit, not an immediate give-up.
func WithDrainTimeout(timeout time.Duration) OptionFunc {
	return func(o *Options) {
		o.DrainTimeout = timeout
	}
}

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	backoff := c.baseDelay
	maxRetries := c.maxRetries
	retries := 0

	for {
		res, err := c.client.Embeddings(ctx, inputs, funcs...)
		if err != nil {
			if retries >= maxRetries {
				return nil, errors.WithStack(err)
			}

			if llm.IsRetryable(err) {
				slog.DebugContext(ctx, "request failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", errors.WithStack(err)))

				retries++
				if !wait(ctx, backoff) {
					// Both stay visible: the caller left, and why it was waiting.
					return nil, errors.WithStack(stderrors.Join(ctx.Err(), err))
				}
				backoff *= 2
				continue
			}

			return nil, errors.WithStack(err)
		}

		return res, nil
	}
}

// Decision implements llm.DecisionClient. An evaluation has no side effect
// upstream, so a failed one is replayed like an embeddings call.
func (c *Client) Decision(ctx context.Context, state any, questions llm.Questions, funcs ...llm.DecisionOptionFunc) (llm.DecisionResponse, error) {
	if !llm.SupportsDecision(c.client) {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}
	backoff := c.baseDelay
	retries := 0

	for {
		res, err := llm.Decide(ctx, c.client, state, questions, funcs...)
		if err == nil {
			return res, nil
		}
		if retries >= c.maxRetries || !llm.IsRetryable(err) {
			return nil, errors.WithStack(err)
		}
		slog.DebugContext(ctx, "decision failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", err))
		retries++
		if !wait(ctx, backoff) {
			return nil, errors.WithStack(stderrors.Join(ctx.Err(), err))
		}
		backoff *= 2
	}
}

// SupportsDecision reports on the wrapped client, see llm.SupportsDecision.
func (c *Client) SupportsDecision() bool {
	return llm.SupportsDecision(c.client)
}

// Transcription implements llm.Client.
func (c *Client) Transcription(ctx context.Context, audio []byte, funcs ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	backoff := c.baseDelay
	maxRetries := c.maxRetries
	retries := 0

	for {
		res, err := c.client.Transcription(ctx, audio, funcs...)
		if err != nil {
			if retries >= maxRetries {
				return nil, errors.WithStack(err)
			}

			if llm.IsRetryable(err) {
				slog.DebugContext(ctx, "request failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", errors.WithStack(err)))

				retries++
				if !wait(ctx, backoff) {
					// Both stay visible: the caller left, and why it was waiting.
					return nil, errors.WithStack(stderrors.Join(ctx.Err(), err))
				}
				backoff *= 2
				continue
			}

			return nil, errors.WithStack(err)
		}

		return res, nil
	}
}

// ChatCompletion implements llm.ChatCompletionClient.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	backoff := c.baseDelay
	maxRetries := c.maxRetries
	retries := 0

	for {
		res, err := c.client.ChatCompletion(ctx, funcs...)
		if err != nil {
			if retries >= maxRetries {
				return nil, errors.WithStack(err)
			}

			if llm.IsRetryable(err) {
				slog.DebugContext(ctx, "request failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", errors.WithStack(err)))

				retries++
				if !wait(ctx, backoff) {
					// Both stay visible: the caller left, and why it was waiting.
					return nil, errors.WithStack(stderrors.Join(ctx.Err(), err))
				}
				backoff *= 2
				continue
			}

			return nil, errors.WithStack(err)
		}

		return res, nil
	}
}

// ChatCompletionStream implements llm.Client.
// Stream errors that are retryable (e.g. 429) trigger a full retry of the call,
// but only while nothing has been forwarded: a retried call starts its answer
// over, and the consumer would get the chunks already sent a second time,
// glued to a different answer. An error after the first forwarded chunk ends
// the stream, as a non-retryable one does.
// All retries and stream reading happen inside a goroutine; the returned channel
// carries both data chunks and any eventual non-retryable error chunk.
func (c *Client) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	outCh := make(chan llm.StreamChunk, 10)

	// See llm.SendChunk: forwarding with a bare channel write would strand this
	// goroutine, and the stream it wraps, as soon as a consumer stops reading.
	// The chunk that ends the stream goes through llm.SendTerminalChunk, which
	// survives the cancellation it usually reports.
	send := func(chunk llm.StreamChunk) bool { return llm.SendChunk(ctx, outCh, chunk) }
	sendTerminal := func(chunk llm.StreamChunk) { llm.SendTerminalChunk(ctx, outCh, chunk) }

	go func() {
		defer close(outCh)

		backoff := c.baseDelay
		retries := 0

		// Each attempt gets its own context so that the stream of an abandoned
		// one can be told to stop. Without it the provider goroutine of a
		// retried attempt stays blocked on a send nobody reads — ctx is still
		// alive — holding its upstream response open, and billed, for good.
		var cancelAttempt context.CancelFunc
		abandonAttempt := func(stream <-chan llm.StreamChunk) {
			if cancelAttempt == nil {
				return
			}
			cancelAttempt()
			cancelAttempt = nil
			go func() {
				if !llm.DrainStream(stream, c.drainTimeout) {
					slog.WarnContext(ctx, "gave up draining an abandoned attempt",
						slog.Duration("after", c.drainTimeout))
				}
			}()
		}

		for {
			attemptCtx, cancel := context.WithCancel(ctx)
			cancelAttempt = cancel
			stream, err := c.client.ChatCompletionStream(attemptCtx, funcs...)
			if err != nil {
				// Nothing was opened, but the attempt context still has to be
				// released before the next one replaces it.
				cancelAttempt()
				cancelAttempt = nil
				if retries < c.maxRetries && llm.IsRetryable(err) {
					slog.DebugContext(ctx, "stream open failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", err))
					retries++
					if !wait(ctx, backoff) {
						sendTerminal(llm.NewErrorStreamChunk(errors.WithStack(stderrors.Join(ctx.Err(), err))))
						return
					}
					backoff *= 2
					continue
				}
				sendTerminal(llm.NewErrorStreamChunk(errors.WithStack(err)))
				return
			}

			// Read the stream; retry the whole call on retryable chunk errors.
			// Use select to respect ctx cancellation — a stalled TCP connection
			// would otherwise block this goroutine indefinitely.
			var retryCall bool
			// retried is the error the next attempt is retrying.
			var retried error
			// forwarded records that this attempt already sent output on: from
			// then on, a retry would repeat it. A chunk carrying nothing, such as
			// the role-only delta streams often open with, leaves it unset.
			forwarded := false
		streamLoop:
			for {
				select {
				case chunk, ok := <-stream:
					if !ok {
						break streamLoop
					}
					if chunkErr := chunk.Error(); chunkErr != nil {
						if !forwarded && retries < c.maxRetries && llm.IsRetryable(chunkErr) {
							slog.DebugContext(ctx, "stream chunk error, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", chunkErr))
							retries++
							retryCall = true
							retried = chunkErr
							break streamLoop
						}
						// Not retryable, out of retries, or too late to retry:
						// forward and stop.
						abandonAttempt(stream)
						sendTerminal(chunk)
						return
					}
					if !send(chunk) {
						// The consumer is gone. Say why the stream ends rather
						// than closing the channel on nothing, which reads as a
						// truncation — an upstream incident — instead of the
						// ordinary hangup it is.
						abandonAttempt(stream)
						sendTerminal(llm.NewErrorStreamChunk(errors.WithStack(ctx.Err())))
						return
					}
					// A completion chunk counts too: a provider sending anything
					// after it breaks the contract, and a retry would follow a
					// whole answer with a second one.
					forwarded = forwarded || chunk.IsComplete() || carriesOutput(chunk)
				case <-ctx.Done():
					abandonAttempt(stream)
					sendTerminal(llm.NewErrorStreamChunk(errors.WithStack(ctx.Err())))
					return
				}
			}

			if !retryCall {
				// The provider closed its channel: nothing left to abandon,
				// only the attempt context to release.
				cancelAttempt()
				return // stream ended normally
			}
			abandonAttempt(stream)
			// retryCall == true: wait (respecting ctx) then open a fresh stream
			if !wait(ctx, backoff) {
				sendTerminal(llm.NewErrorStreamChunk(errors.WithStack(stderrors.Join(ctx.Err(), retried))))
				return
			}
			backoff *= 2
		}
	}()

	return outCh, nil
}

// RelayMessages implements llm.MessagesRelayClient. Only opening the stream is
// retried: once an event has been relayed, replaying the request would send
// the client a second copy of the message, which the Messages protocol has no
// way to take back.
func (c *Client) RelayMessages(ctx context.Context, body []byte, header http.Header) (<-chan llm.StreamChunk, error) {
	backoff := c.baseDelay
	retries := 0

	for {
		stream, err := llm.RelayMessages(ctx, c.client, body, header)
		if err == nil {
			return stream, nil
		}
		if retries >= c.maxRetries || !llm.IsRetryable(err) {
			return nil, errors.WithStack(err)
		}
		slog.DebugContext(ctx, "relay failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", err))
		retries++
		if !wait(ctx, backoff) {
			return nil, errors.WithStack(stderrors.Join(ctx.Err(), err))
		}
		backoff *= 2
	}
}

// SupportsMessagesRelay reports on the wrapped client, see
// llm.SupportsMessagesRelay.
func (c *Client) SupportsMessagesRelay() bool {
	return llm.SupportsMessagesRelay(c.client)
}

func NewClient(client llm.Client, baseDelay time.Duration, maxRetries int, funcs ...OptionFunc) *Client {
	opts := &Options{DrainTimeout: DefaultDrainTimeout}
	for _, fn := range funcs {
		fn(opts)
	}
	return &Client{
		baseDelay:    baseDelay,
		maxRetries:   maxRetries,
		drainTimeout: opts.DrainTimeout,
		client:       client,
	}
}

var (
	_ llm.Client              = &Client{}
	_ llm.MessagesRelayClient = &Client{}
	_ llm.DecisionClient      = &Client{}
)

// Close releases the wrapped client when it implements io.Closer, so that a
// plugin client stays releasable behind this decorator. The decorator owns
// the client it wraps: closing it closes that client for every other holder
// too, and an implementation of Close must tolerate being called twice.
func (c *Client) Close() error {
	if closer, ok := c.client.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}

// carriesOutput reports whether chunk holds output a consumer shows or acts
// on: content, tool calls, reasoning or audio.
func carriesOutput(chunk llm.StreamChunk) bool {
	delta := chunk.Delta()
	if delta == nil {
		return false
	}
	if delta.Content() != "" || len(delta.ToolCalls()) > 0 {
		return true
	}
	if r, ok := delta.(llm.ReasoningStreamDelta); ok && (r.Reasoning() != "" || len(r.ReasoningDetails()) > 0) {
		return true
	}
	type audioDelta interface {
		AudioData() string
		Transcript() string
	}
	if a, ok := delta.(audioDelta); ok && (a.AudioData() != "" || a.Transcript() != "") {
		return true
	}
	return false
}

// wait waits for d, or until ctx ends, and reports whether d elapsed. Unlike
// time.After, its timer is released when ctx wins.
func wait(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}

package proxy

import (
	"context"
	"errors"
	"io"
	"net/http"

	"github.com/bornholm/genai/llm"
)

// messagesRelayStream presents a relayed Messages request as a streaming
// client, so that it goes through streamChatCompletion like any other stream
// and gets the same error handling, interruption tracking and post-response
// hooks. The chat completion options are ignored: the request is the body.
type messagesRelayStream struct {
	client llm.MessagesRelayClient
	body   []byte
	header http.Header
}

func (s *messagesRelayStream) ChatCompletionStream(ctx context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	return s.client.RelayMessages(ctx, s.body, s.header)
}

// rawEventEmitter writes relayed events through unchanged. It writes no
// closing events of its own: the upstream's are among the events, and making
// some up for a truncated stream would pass it off as complete to a client
// that knows to retry one that is not.
type rawEventEmitter struct{}

func (rawEventEmitter) EmitFirst(w io.Writer, chunk llm.StreamChunk) error {
	return rawEventEmitter{}.Emit(w, chunk)
}

func (rawEventEmitter) Emit(w io.Writer, chunk llm.StreamChunk) error {
	raw, ok := chunk.(llm.RawEventChunk)
	if !ok {
		return errors.New("relay: stream chunk carries no raw event")
	}
	_, err := w.Write(raw.RawEvent())
	return err
}

// EmitError writes an upstream error event as received, and makes up an
// Anthropic error event for a failure of the relay itself.
func (rawEventEmitter) EmitError(w io.Writer, err error) error {
	var rawErr *llm.RawEventError
	if errors.As(err, &rawErr) {
		_, writeErr := w.Write(rawErr.Event)
		return writeErr
	}
	return (&anthropicStreamEmitter{}).EmitError(w, err)
}

func (rawEventEmitter) Finalize(io.Writer, llm.ChatCompletionUsage) error {
	return nil
}

// relayedErrorHeaders are the upstream headers a rejection keeps: those a
// client retries on. The rest describe the operator's upstream account
// (organization, rate limits) or its infrastructure, not the client's call.
var relayedErrorHeaders = []string{"Content-Type", "Retry-After", "X-Should-Retry", "Request-Id"}

// WriteErrorResponse forwards an upstream rejection as received: status,
// retry headers and body. Clients recognize some rejections by their wording
// and recover from them, which a rewritten error defeats. An error event
// arriving first in the stream is answered here too, as an HTTP error rather
// than an SSE event: the SSE headers are not committed yet, and a status is
// what clients retry on.
func (rawEventEmitter) WriteErrorResponse(w http.ResponseWriter, err error) {
	var httpErr *llm.HTTPError
	if !errors.As(err, &httpErr) {
		writeAnthropicAPIError(w, apiErrorFromErr(err))
		return
	}
	for _, name := range relayedErrorHeaders {
		if value := httpErr.Header.Get(name); value != "" {
			w.Header().Set(name, value)
		}
	}
	if w.Header().Get("Content-Type") == "" {
		w.Header().Set("Content-Type", "application/json")
	}
	w.WriteHeader(httpErr.StatusCode)
	_, _ = io.WriteString(w, httpErr.Body)
}

var _ errorResponseWriter = rawEventEmitter{}

package llm

import (
	"context"
	"net/http"
)

// MessagesRelayClient is implemented by the chat completion clients that speak
// the Anthropic Messages API natively and can forward a request in that format
// to their upstream as is, instead of rebuilding it from ChatCompletionOptions.
//
// Relaying keeps what a translation loses: the request fields and the
// anthropic-beta values genai has no option for, and the response fields its
// stream chunks have no room for. Each event of the upstream stream comes back
// verbatim as a RawEventChunk.
//
// Only streamed requests are relayed.
type MessagesRelayClient interface {
	// RelayMessages posts body, an Anthropic Messages request with "stream":
	// true, along with the anthropic-* headers found in header; every other
	// header is ignored. The client substitutes its own credentials and, when
	// it is bound to a model, its model. A non-2xx answer is returned as an
	// error wrapping *HTTPError, before any chunk.
	RelayMessages(ctx context.Context, body []byte, header http.Header) (<-chan StreamChunk, error)
}

// SupportsMessagesRelay reports whether client can relay Messages requests.
//
// A decorator has a RelayMessages method whatever it wraps, so it also
// implements SupportsMessagesRelay() bool to report on the client it wraps.
// A decorator without either method hides the relay: requests then go
// through the translated path, which works, only without what the relay
// keeps.
func SupportsMessagesRelay(client any) bool {
	if _, ok := client.(MessagesRelayClient); !ok {
		return false
	}
	if s, ok := client.(interface{ SupportsMessagesRelay() bool }); ok {
		return s.SupportsMessagesRelay()
	}
	return true
}

// RelayMessages relays through client, or returns ErrUnavailable when it
// cannot. It is how a decorator delegates to the client it wraps.
func RelayMessages(ctx context.Context, client any, body []byte, header http.Header) (<-chan StreamChunk, error) {
	if !SupportsMessagesRelay(client) {
		return nil, ErrUnavailable
	}
	return client.(MessagesRelayClient).RelayMessages(ctx, body, header)
}

// RawEventChunk is a StreamChunk carrying one server-sent event exactly as the
// upstream sent it, blank line terminator included. Delta is always nil: the
// content lives in the event, which consumers write through unchanged. Usage
// is the cumulative usage the stream published up to this event.
type RawEventChunk interface {
	StreamChunk
	RawEvent() []byte
}

// RawEventError is the error of a chunk relaying an upstream error event,
// which Event holds verbatim so that it can be written through as is.
type RawEventError struct {
	Event []byte
	Err   error
}

func (e *RawEventError) Error() string { return e.Err.Error() }
func (e *RawEventError) Unwrap() error { return e.Err }

// NewRawEventChunk returns a chunk relaying event. complete marks the event
// that ends the stream (message_stop).
func NewRawEventChunk(event []byte, usage ChatCompletionUsage, complete bool) RawEventChunk {
	return &rawEventChunk{event: event, usage: usage, complete: complete}
}

// NewRawEventErrorChunk returns a chunk relaying an upstream error event.
func NewRawEventErrorChunk(event []byte, err error, usage ChatCompletionUsage) RawEventChunk {
	return &rawEventChunk{event: event, usage: usage, err: &RawEventError{Event: event, Err: err}}
}

type rawEventChunk struct {
	event    []byte
	usage    ChatCompletionUsage
	complete bool
	err      error
}

func (c *rawEventChunk) Type() StreamChunkType {
	switch {
	case c.err != nil:
		return StreamChunkTypeError
	case c.complete:
		return StreamChunkTypeComplete
	default:
		return StreamChunkTypeDelta
	}
}

func (c *rawEventChunk) Delta() StreamDelta         { return nil }
func (c *rawEventChunk) Usage() ChatCompletionUsage { return c.usage }
func (c *rawEventChunk) Error() error               { return c.err }
func (c *rawEventChunk) IsComplete() bool           { return c.complete }
func (c *rawEventChunk) RawEvent() []byte           { return c.event }

var _ RawEventChunk = &rawEventChunk{}

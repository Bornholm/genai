package proxy

import (
	"bytes"
	"time"

	"github.com/bornholm/genai/llm"
)

// ExchangeTiming records when an exchange reached each of its stages. The
// handler fills it as the request goes along, so a hook reads whatever stages
// were reached by the time it runs: a post-response hook sees them all, an
// error hook sees those reached before the failure.
//
// A zero time means the stage was not reached, or does not apply. The
// durations below report false in that case rather than a meaningless value.
type ExchangeTiming struct {
	// ReceivedAt is when the request body had been read, before
	// authentication and the pre-request hooks. The time spent uploading the
	// body is left out: it depends on the client's link, not on the gateway.
	ReceivedAt time.Time
	// UpstreamStartAt is when the call to the resolved client started, once
	// the pre-request hooks had run and the model had been resolved. Zero when
	// the request never got that far.
	UpstreamStartAt time.Time
	// FirstOutputAt is when the first streamed chunk carrying output reached
	// the proxy: text, reasoning or a tool call. A chunk announcing the role
	// alone does not count. Zero for a response that was not streamed, and for
	// a stream that ended before producing any output.
	//
	// It is measured where the proxy reads the stream, so it includes whatever
	// the resolved client does on the way, a decorator buffering the answer
	// included: that is when the client could have received it.
	FirstOutputAt time.Time
	// CompletedAt is when the upstream call ended: the response returned, the
	// stream stopped being read, or the call failed. Zero when the upstream was
	// never called.
	CompletedAt time.Time
}

// RoutingDuration is the time the proxy spent before calling the upstream:
// authentication, pre-request hooks and model resolution.
func (t ExchangeTiming) RoutingDuration() (time.Duration, bool) {
	return between(t.ReceivedAt, t.UpstreamStartAt)
}

// TimeToFirstOutput is the time between the upstream call and the first chunk
// carrying output, the latency a streaming client perceives from the provider.
func (t ExchangeTiming) TimeToFirstOutput() (time.Duration, bool) {
	return between(t.UpstreamStartAt, t.FirstOutputAt)
}

// GenerationDuration is the time between the first chunk carrying output and
// the end of the stream. Output tokens divided by it give the throughput.
func (t ExchangeTiming) GenerationDuration() (time.Duration, bool) {
	return between(t.FirstOutputAt, t.CompletedAt)
}

// UpstreamDuration is the time the upstream call took, from its start to its
// end, whether it succeeded or not.
func (t ExchangeTiming) UpstreamDuration() (time.Duration, bool) {
	return between(t.UpstreamStartAt, t.CompletedAt)
}

// TotalDuration is the time between the request being received and the
// upstream call ending.
func (t ExchangeTiming) TotalDuration() (time.Duration, bool) {
	return between(t.ReceivedAt, t.CompletedAt)
}

func between(from, to time.Time) (time.Duration, bool) {
	if from.IsZero() || to.IsZero() || to.Before(from) {
		return 0, false
	}
	return to.Sub(from), true
}

// anthropicContentEvent is the type of the only relayed Messages events that
// carry output: text, thinking and tool input all arrive as deltas of a content
// block. It is looked for anywhere in the event, since the SSE event line is
// optional and the type is repeated in the data; no other event of a Messages
// stream carries the name.
var anthropicContentEvent = []byte("content_block_delta")

// carriesOutput reports whether a stream chunk carries part of the answer, as
// opposed to a chunk announcing the role, reporting usage or closing the
// stream.
func carriesOutput(chunk llm.StreamChunk) bool {
	if chunk == nil || chunk.Error() != nil {
		return false
	}
	if raw, ok := chunk.(llm.RawEventChunk); ok {
		return bytes.Contains(raw.RawEvent(), anthropicContentEvent)
	}
	delta := chunk.Delta()
	if delta == nil {
		return false
	}
	if delta.Content() != "" || len(delta.ToolCalls()) > 0 {
		return true
	}
	if rd, ok := delta.(llm.ReasoningStreamDelta); ok {
		return rd.Reasoning() != "" || len(rd.ReasoningDetails()) > 0
	}
	return false
}

package proxy

import (
	"bytes"
	"encoding/json"
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

// carriesOutput reports whether a stream chunk carries part of the answer, as
// opposed to a chunk announcing the role, reporting usage or closing the
// stream.
func carriesOutput(chunk llm.StreamChunk) bool {
	if chunk == nil || chunk.Error() != nil {
		return false
	}
	if raw, ok := chunk.(llm.RawEventChunk); ok {
		return messagesEventCarriesOutput(raw.RawEvent())
	}
	return llm.ChunkCarriesOutput(chunk)
}

// messagesEvent is a relayed Messages event, reduced to what tells whether it
// carries output.
type messagesEvent struct {
	Type         string `json:"type"`
	ContentBlock struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		Thinking string `json:"thinking"`
	} `json:"content_block"`
}

// messagesEventCarriesOutput reports whether a relayed Messages event carries
// output: a content block delta, or the start of a block that already holds
// some. A tool_use block counts from its start, which names the tool, as a
// tool call delta naming it does on the translated path; a redacted thinking
// block arrives whole in its start event. The data lines of the event are
// joined with newlines before decoding, as SSE specifies. An event whose data
// does not decode carries none.
func messagesEventCarriesOutput(event []byte) bool {
	var data [][]byte
	for _, line := range bytes.Split(event, []byte("\n")) {
		if value, ok := bytes.CutPrefix(bytes.TrimRight(line, "\r"), []byte("data:")); ok {
			data = append(data, bytes.TrimPrefix(value, []byte(" ")))
		}
	}
	var parsed messagesEvent
	if json.Unmarshal(bytes.Join(data, []byte("\n")), &parsed) != nil {
		return false
	}
	switch parsed.Type {
	case "content_block_delta":
		return true
	case "content_block_start":
		block := parsed.ContentBlock
		return block.Type == "tool_use" || block.Type == "redacted_thinking" || block.Text != "" || block.Thinking != ""
	default:
		return false
	}
}

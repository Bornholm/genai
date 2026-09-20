package llm

import (
	"context"
	"time"
)

// ChatCompletionStreamingClient defines the interface for streaming chat completions.
//
// An implementation owes its consumer two things. It must end the stream with a
// terminal chunk — a complete chunk or an error chunk — because that is how a
// consumer tells a finished response from one cut short. And it must honour the
// cancellation of ctx while sending, by selecting on ctx.Done() rather than
// writing to the channel bare: a consumer is free to stop reading at any point,
// and an implementation that then blocks forever never runs its own cleanup,
// keeping whatever connection it holds open.
type ChatCompletionStreamingClient interface {
	ChatCompletionStream(ctx context.Context, funcs ...ChatCompletionOptionFunc) (<-chan StreamChunk, error)
}

// SendChunk writes one chunk to a provider's channel, giving up if the consumer
// cancelled ctx. It is what an implementation of ChatCompletionStreamingClient
// owes its consumer on every delta: a bare channel write strands the producing
// goroutine — and whatever connection it holds — the moment a consumer stops
// reading. It reports whether the chunk was delivered.
func SendChunk(ctx context.Context, chunks chan<- StreamChunk, chunk StreamChunk) bool {
	select {
	case chunks <- chunk:
		return true
	case <-ctx.Done():
		return false
	}
}

// SendTerminalChunk writes the chunk that ends a stream — a completion chunk or
// an error chunk — and tries harder than SendChunk to deliver it.
//
// Cancellation is often the very thing a terminal chunk reports, and by then
// ctx.Done() is closed: a plain select would have both of its cases ready and
// the runtime would pick between them at random, dropping the chunk about half
// the time and leaving the consumer with a silently truncated stream instead of
// the error. Taking the buffer slot first avoids that coin flip; a full — or
// unbuffered — channel falls back to SendChunk, which on a canceled context may
// well give up, so the guarantee only holds for a channel with room to spare.
// It reports whether the chunk was delivered.
func SendTerminalChunk(ctx context.Context, chunks chan<- StreamChunk, chunk StreamChunk) bool {
	select {
	case chunks <- chunk:
		return true
	default:
		return SendChunk(ctx, chunks, chunk)
	}
}

// DrainStream reads and discards what is left of a stream whose consumer has
// given up, so that the goroutine producing it reaches its own cleanup instead
// of blocking forever on a channel nobody reads — with, typically, an upstream
// HTTP response held open and billed behind it.
//
// Cancelling the producer's context is the first thing to do and is not enough:
// an implementation whose sends do not watch it stays blocked regardless. The
// timeout caps what such an implementation can hold; zero or less means no cap.
// It reports whether the stream ended on its own before the deadline.
func DrainStream(stream <-chan StreamChunk, timeout time.Duration) bool {
	var expired <-chan time.Time
	if timeout > 0 {
		timer := time.NewTimer(timeout)
		defer timer.Stop()
		expired = timer.C
	}
	for {
		select {
		case _, ok := <-stream:
			if !ok {
				return true
			}
		case <-expired:
			return false
		}
	}
}

// StreamChunkType represents the type of streaming chunk
type StreamChunkType string

const (
	StreamChunkTypeDelta    StreamChunkType = "delta"
	StreamChunkTypeUsage    StreamChunkType = "usage"
	StreamChunkTypeError    StreamChunkType = "error"
	StreamChunkTypeComplete StreamChunkType = "complete"
)

// StreamChunk represents a single streaming response chunk
type StreamChunk interface {
	Type() StreamChunkType
	Delta() StreamDelta
	Usage() ChatCompletionUsage
	Error() error
	IsComplete() bool
}

// StreamDelta represents incremental content changes in a streaming response
type StreamDelta interface {
	Role() Role
	Content() string
	ToolCalls() []ToolCallDelta
}

// ReasoningStreamDelta extends StreamDelta with reasoning content.
// Callers can type-assert a StreamDelta to this interface to access
// incremental reasoning tokens emitted during streaming.
type ReasoningStreamDelta interface {
	StreamDelta
	// Reasoning returns the incremental reasoning text in this chunk.
	Reasoning() string
	// ReasoningDetails returns incremental reasoning detail blocks in this chunk.
	ReasoningDetails() []ReasoningDetail
}

// ToolCallDelta represents incremental tool call data in streaming
type ToolCallDelta interface {
	Index() int
	ID() string
	Name() string
	ParametersDelta() string
}

// BaseStreamChunk provides a base implementation of StreamChunk
type BaseStreamChunk struct {
	chunkType StreamChunkType
	delta     StreamDelta
	usage     ChatCompletionUsage
	err       error
	complete  bool
}

// Type implements StreamChunk
func (c *BaseStreamChunk) Type() StreamChunkType {
	return c.chunkType
}

// Delta implements StreamChunk
func (c *BaseStreamChunk) Delta() StreamDelta {
	return c.delta
}

// Usage implements StreamChunk
func (c *BaseStreamChunk) Usage() ChatCompletionUsage {
	return c.usage
}

// Error implements StreamChunk
func (c *BaseStreamChunk) Error() error {
	return c.err
}

// IsComplete implements StreamChunk
func (c *BaseStreamChunk) IsComplete() bool {
	return c.complete
}

var _ StreamChunk = &BaseStreamChunk{}

// BaseStreamDelta provides a base implementation of StreamDelta and ReasoningStreamDelta
type BaseStreamDelta struct {
	role             Role
	content          string
	toolCalls        []ToolCallDelta
	reasoning        string
	reasoningDetails []ReasoningDetail
	audioData        string
	transcript       string
}

// Role implements StreamDelta
func (d *BaseStreamDelta) Role() Role {
	return d.role
}

// Content implements StreamDelta
func (d *BaseStreamDelta) Content() string {
	return d.content
}

// ToolCalls implements StreamDelta
func (d *BaseStreamDelta) ToolCalls() []ToolCallDelta {
	return d.toolCalls
}

// Reasoning implements ReasoningStreamDelta
func (d *BaseStreamDelta) Reasoning() string {
	return d.reasoning
}

// ReasoningDetails implements ReasoningStreamDelta
func (d *BaseStreamDelta) ReasoningDetails() []ReasoningDetail {
	return d.reasoningDetails
}

// AudioData returns the base64-encoded audio data in this chunk
func (d *BaseStreamDelta) AudioData() string {
	return d.audioData
}

// Transcript returns the transcript in this audio chunk
func (d *BaseStreamDelta) Transcript() string {
	return d.transcript
}

var _ StreamDelta = &BaseStreamDelta{}
var _ ReasoningStreamDelta = &BaseStreamDelta{}

// BaseToolCallDelta provides a base implementation of ToolCallDelta
type BaseToolCallDelta struct {
	index           int
	id              string
	name            string
	parametersDelta string
}

// Index implements ToolCallDelta
func (t *BaseToolCallDelta) Index() int {
	return t.index
}

// ID implements ToolCallDelta
func (t *BaseToolCallDelta) ID() string {
	return t.id
}

// Name implements ToolCallDelta
func (t *BaseToolCallDelta) Name() string {
	return t.name
}

// ParametersDelta implements ToolCallDelta
func (t *BaseToolCallDelta) ParametersDelta() string {
	return t.parametersDelta
}

var _ ToolCallDelta = &BaseToolCallDelta{}

// NewStreamChunk creates a new streaming chunk with delta content
func NewStreamChunk(delta StreamDelta) *BaseStreamChunk {
	return &BaseStreamChunk{
		chunkType: StreamChunkTypeDelta,
		delta:     delta,
		complete:  false,
	}
}

// NewStreamChunkWithUsage creates a delta chunk that also carries the usage
// accounted for so far.
//
// Providers that learn their counters while the stream runs — Anthropic reports
// the input tokens in message_start, OpenAI-compatible gateways sometimes send
// usage mid-stream — should use it instead of NewStreamChunk. A consumer whose
// stream is cut short before the final chunk then still has the counts the
// provider had published at that point, instead of zeroes. The usage is
// cumulative, not incremental: each chunk carries the totals known so far, which
// is what StreamingUsageTracker expects.
func NewStreamChunkWithUsage(delta StreamDelta, usage ChatCompletionUsage) *BaseStreamChunk {
	return &BaseStreamChunk{
		chunkType: StreamChunkTypeDelta,
		delta:     delta,
		usage:     usage,
		complete:  false,
	}
}

// NewCompleteStreamChunk creates a final streaming chunk with usage information
func NewCompleteStreamChunk(usage ChatCompletionUsage) *BaseStreamChunk {
	return &BaseStreamChunk{
		chunkType: StreamChunkTypeComplete,
		usage:     usage,
		complete:  true,
	}
}

// NewErrorStreamChunk creates an error streaming chunk
func NewErrorStreamChunk(err error) *BaseStreamChunk {
	return &BaseStreamChunk{
		chunkType: StreamChunkTypeError,
		err:       err,
		complete:  false,
	}
}

// NewErrorStreamChunkWithUsage creates an error streaming chunk that also
// carries the usage accounted for before the failure. The provider billed what
// it had already produced, so a caller recording usage keeps it even though the
// stream never reached its completion chunk.
func NewErrorStreamChunkWithUsage(err error, usage ChatCompletionUsage) *BaseStreamChunk {
	return &BaseStreamChunk{
		chunkType: StreamChunkTypeError,
		err:       err,
		usage:     usage,
		complete:  false,
	}
}

// NewStreamDelta creates a new stream delta
func NewStreamDelta(role Role, content string, toolCalls ...ToolCallDelta) *BaseStreamDelta {
	return &BaseStreamDelta{
		role:      role,
		content:   content,
		toolCalls: toolCalls,
	}
}

// NewReasoningStreamDelta creates a new stream delta that also carries incremental
// reasoning content. Use this when the provider returns reasoning tokens during streaming.
func NewReasoningStreamDelta(role Role, content string, reasoning string, reasoningDetails []ReasoningDetail, toolCalls ...ToolCallDelta) *BaseStreamDelta {
	return &BaseStreamDelta{
		role:             role,
		content:          content,
		toolCalls:        toolCalls,
		reasoning:        reasoning,
		reasoningDetails: reasoningDetails,
	}
}

// NewAudioStreamDelta creates a new stream delta that also carries audio data.
// Use this when the provider returns audio chunks during streaming (e.g., text-to-audio models).
func NewAudioStreamDelta(role Role, content, audioData, transcript string, toolCalls ...ToolCallDelta) *BaseStreamDelta {
	return &BaseStreamDelta{
		role:       role,
		content:    content,
		toolCalls:  toolCalls,
		audioData:  audioData,
		transcript: transcript,
	}
}

// SetStreamDeltaReasoning sets the reasoning of a delta built by another
// constructor, for a delta carrying both audio and reasoning.
func SetStreamDeltaReasoning(delta *BaseStreamDelta, reasoning string, details []ReasoningDetail) {
	delta.reasoning = reasoning
	delta.reasoningDetails = details
}

// NewToolCallDelta creates a new tool call delta
func NewToolCallDelta(index int, id, name, parametersDelta string) *BaseToolCallDelta {
	return &BaseToolCallDelta{
		index:           index,
		id:              id,
		name:            name,
		parametersDelta: parametersDelta,
	}
}

// StreamingUsageTracker tracks token usage across streaming chunks
type StreamingUsageTracker struct {
	promptTokens        int64
	completionTokens    int64
	totalTokens         int64
	cachedTokens        int64
	cacheCreationTokens int64
	cost                *float64
	costCurrency        string
	reported            bool
}

// Update updates the usage tracker with data from a streaming chunk
func (t *StreamingUsageTracker) Update(chunk StreamChunk) {
	if usage := chunk.Usage(); usage != nil {
		// An all-zero usage is not a report, and is ignored outright. Providers
		// synthesize one to carry a termination signal and a gateway may return
		// an empty usage object; counting it would make Reported() claim
		// counters nobody published — the confusion between "zero" and
		// "unknown" that flag exists to prevent — and letting it through would
		// wipe counters a previous chunk did publish.
		if !UsagePublishesCounters(usage) {
			return
		}
		t.reported = true
		t.promptTokens = usage.PromptTokens()
		t.completionTokens = usage.CompletionTokens()
		t.totalTokens = usage.TotalTokens()
		type cachedUsage interface{ CachedTokens() int64 }
		if cu, ok := usage.(cachedUsage); ok {
			t.cachedTokens = cu.CachedTokens()
		}
		if cc, ok := usage.(CacheCreationReportingUsage); ok {
			t.cacheCreationTokens = cc.CacheCreationTokens()
		}
		if cr, ok := usage.(CostReportingUsage); ok {
			if amount, currency, ok := cr.Cost(); ok {
				t.cost = &amount
				t.costCurrency = currency
			}
		}
	}
}

// UsagePublishesCounters reports whether a usage carries a measurement worth
// recording, as opposed to an empty shell.
//
// It is the one definition of "the provider reported something": an all-zero
// usage is what a provider sends to signal the end of a stream, and what a
// gateway returns when it has nothing to say, so counting it would turn
// "unknown" into "zero" everywhere that distinction matters. Cache counters and
// a reported cost count: a cached prompt, or a priced call, is a measurement
// even when no token counter came with it.
func UsagePublishesCounters(usage ChatCompletionUsage) bool {
	if usage == nil {
		return false
	}
	if usage.PromptTokens() > 0 || usage.CompletionTokens() > 0 || usage.TotalTokens() > 0 {
		return true
	}
	type cachedUsage interface{ CachedTokens() int64 }
	if cu, ok := usage.(cachedUsage); ok && cu.CachedTokens() > 0 {
		return true
	}
	if cc, ok := usage.(CacheCreationReportingUsage); ok && cc.CacheCreationTokens() > 0 {
		return true
	}
	if cr, ok := usage.(CostReportingUsage); ok {
		if amount, _, ok := cr.Cost(); ok && amount > 0 {
			return true
		}
	}
	return false
}

// Reported reports whether any chunk carried usage at all. It tells apart a
// provider that published zero tokens from one that published nothing, which
// matters when a stream is cut short before its final chunk: the counts are
// then unknown rather than null.
func (t *StreamingUsageTracker) Reported() bool {
	return t.reported
}

// Usage returns the current usage as a ChatCompletionUsage
func (t *StreamingUsageTracker) Usage() ChatCompletionUsage {
	return NewChatCompletionUsageFull(t.promptTokens, t.completionTokens, t.totalTokens, t.cachedTokens, t.cacheCreationTokens, t.cost, t.costCurrency)
}

// NewStreamingUsageTracker creates a new streaming usage tracker
func NewStreamingUsageTracker() *StreamingUsageTracker {
	return &StreamingUsageTracker{}
}

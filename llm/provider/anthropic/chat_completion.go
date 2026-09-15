package anthropic

import (
	"context"
	"net/http"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// ChatCompletionClient talks to the Anthropic Messages API.
//
// Both entry points go through the streaming endpoint: the SDK refuses a
// non-streaming request whose max_tokens implies more than ten minutes of
// generation, a limit a gateway forwarding client-chosen max_tokens hits
// routinely. Accumulating the stream yields the very same Message.
type ChatCompletionClient struct {
	client    anthropicsdk.Client
	model     string
	maxTokens int64
}

// ChatCompletion implements llm.ChatCompletionClient.
func (c *ChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	params, err := buildParams(opts, c.model, c.maxTokens)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	stream := c.client.Messages.NewStreaming(ctx, *params)
	defer stream.Close()

	message := anthropicsdk.Message{}
	for stream.Next() {
		if err := message.Accumulate(stream.Current()); err != nil {
			return nil, errors.WithStack(err)
		}
	}
	if err := stream.Err(); err != nil {
		return nil, errors.WithStack(mapError(err))
	}

	// A stream that ended cleanly without a single content block is the
	// Messages API counterpart of an empty choices list: report it as
	// ErrNoMessage, which the retry wrapper knows how to handle, rather than
	// as a plausible-looking empty assistant turn.
	if len(message.Content) == 0 {
		return nil, errors.WithStack(llm.ErrNoMessage)
	}

	return fromMessage(&message, excludeReasoning(opts)), nil
}

// ChatCompletionStream implements llm.ChatCompletionStreamingClient.
func (c *ChatCompletionClient) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	params, err := buildParams(opts, c.model, c.maxTokens)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	chunks := make(chan llm.StreamChunk, 10)

	go func() {
		defer close(chunks)

		stream := c.client.Messages.NewStreaming(ctx, *params)
		defer stream.Close()

		emitter := newStreamEmitter(chunks, excludeReasoning(opts))

		for stream.Next() {
			emitter.handle(stream.Current())
		}
		if err := stream.Err(); err != nil {
			chunks <- llm.NewErrorStreamChunk(errors.WithStack(mapError(err)))
			return
		}

		chunks <- llm.NewCompleteStreamChunk(emitter.usage())
	}()

	return chunks, nil
}

// streamEmitter translates Messages API events into genai stream chunks.
//
// Thinking text is streamed incrementally through Reasoning() for display,
// and the complete block (text plus signature) is emitted once as a
// ReasoningDetail when the block closes: a signature only verifies the exact
// text it was computed over, so a detail carrying one without the other
// cannot be replayed.
type streamEmitter struct {
	chunks           chan<- llm.StreamChunk
	excludeReasoning bool

	// toolIndexes maps a content block index to the tool call index genai
	// exposes, which counts tool calls only.
	toolIndexes map[int64]int
	toolCount   int

	// thinking buffers the open thinking blocks by content block index.
	thinking map[int64]*thinkingBlock
	// detailCount numbers the reasoning details in emission order, the
	// same convention fromMessage uses.
	detailCount int

	inputTokens         int64
	outputTokens        int64
	cacheReadTokens     int64
	cacheCreationTokens int64
}

type thinkingBlock struct {
	text      string
	signature string
}

func newStreamEmitter(chunks chan<- llm.StreamChunk, excludeReasoning bool) *streamEmitter {
	return &streamEmitter{
		chunks:           chunks,
		excludeReasoning: excludeReasoning,
		toolIndexes:      map[int64]int{},
		thinking:         map[int64]*thinkingBlock{},
	}
}

// emitDetail sends one complete reasoning detail, numbered in emission order.
func (e *streamEmitter) emitDetail(detail llm.ReasoningDetail) {
	detail.Index = e.detailCount
	e.detailCount++
	e.chunks <- llm.NewStreamChunk(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", "", []llm.ReasoningDetail{detail}))
}

func (e *streamEmitter) handle(event anthropicsdk.MessageStreamEventUnion) {
	switch event.Type {
	case "message_start":
		e.recordUsage(event.Message.Usage)

	case "message_delta":
		e.outputTokens = event.Usage.OutputTokens
		if event.Usage.JSON.InputTokens.Valid() {
			e.inputTokens = event.Usage.InputTokens
		}
		if event.Usage.JSON.CacheReadInputTokens.Valid() {
			e.cacheReadTokens = event.Usage.CacheReadInputTokens
		}
		if event.Usage.JSON.CacheCreationInputTokens.Valid() {
			e.cacheCreationTokens = event.Usage.CacheCreationInputTokens
		}

	case "content_block_start":
		block := event.ContentBlock
		switch block.Type {
		case "text":
			if block.Text != "" {
				e.chunks <- llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, block.Text))
			}
		case "tool_use":
			index := e.toolCount
			e.toolCount++
			e.toolIndexes[event.Index] = index
			e.chunks <- llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "",
				llm.NewToolCallDelta(index, block.ID, block.Name, ""),
			))
		case "thinking":
			e.thinking[event.Index] = &thinkingBlock{text: block.Thinking, signature: block.Signature}
		case "redacted_thinking":
			if e.excludeReasoning {
				return
			}
			e.emitDetail(llm.ReasoningDetail{
				Type: llm.ReasoningDetailTypeEncrypted,
				Data: block.Data,
			})
		}

	case "content_block_stop":
		block, open := e.thinking[event.Index]
		if !open {
			return
		}
		delete(e.thinking, event.Index)
		if e.excludeReasoning {
			return
		}
		e.emitDetail(llm.ReasoningDetail{
			Type:      llm.ReasoningDetailTypeText,
			Text:      block.text,
			Signature: block.signature,
		})

	case "content_block_delta":
		delta := event.Delta
		switch delta.Type {
		case "text_delta":
			e.chunks <- llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, delta.Text))
		case "input_json_delta":
			index, known := e.toolIndexes[event.Index]
			if !known || delta.PartialJSON == "" {
				return
			}
			e.chunks <- llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "",
				llm.NewToolCallDelta(index, "", "", delta.PartialJSON),
			))
		case "thinking_delta":
			if block, open := e.thinking[event.Index]; open {
				block.text += delta.Thinking
			}
			if e.excludeReasoning || delta.Thinking == "" {
				return
			}
			e.chunks <- llm.NewStreamChunk(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", delta.Thinking, nil))
		case "signature_delta":
			if block, open := e.thinking[event.Index]; open {
				block.signature += delta.Signature
			}
		}
	}
}

func (e *streamEmitter) recordUsage(usage anthropicsdk.Usage) {
	e.inputTokens = usage.InputTokens
	e.outputTokens = usage.OutputTokens
	e.cacheReadTokens = usage.CacheReadInputTokens
	e.cacheCreationTokens = usage.CacheCreationInputTokens
}

func (e *streamEmitter) usage() llm.ChatCompletionUsage {
	return newUsage(e.inputTokens, e.outputTokens, e.cacheReadTokens, e.cacheCreationTokens)
}

// fromMessage converts a complete Messages API response.
func fromMessage(message *anthropicsdk.Message, excludeReasoning bool) llm.ChatCompletionResponse {
	var (
		content   string
		reasoning string
		details   []llm.ReasoningDetail
		toolCalls []llm.ToolCall
	)

	for _, block := range message.Content {
		switch block.Type {
		case "text":
			content += block.Text
		case "thinking":
			reasoning += block.Thinking
			details = append(details, llm.ReasoningDetail{
				Type:      llm.ReasoningDetailTypeText,
				Text:      block.Thinking,
				Signature: block.Signature,
				Index:     len(details),
			})
		case "redacted_thinking":
			details = append(details, llm.ReasoningDetail{
				Type:  llm.ReasoningDetailTypeEncrypted,
				Data:  block.Data,
				Index: len(details),
			})
		case "tool_use":
			toolCalls = append(toolCalls, llm.NewToolCall(block.ID, block.Name, string(block.Input)))
		}
	}

	usage := newUsage(
		message.Usage.InputTokens,
		message.Usage.OutputTokens,
		message.Usage.CacheReadInputTokens,
		message.Usage.CacheCreationInputTokens,
	)

	if excludeReasoning || (reasoning == "" && len(details) == 0) {
		return llm.NewChatCompletionResponse(llm.NewMessage(llm.RoleAssistant, content), usage, toolCalls...)
	}

	return llm.NewChatCompletionResponseWithReasoning(
		llm.NewAssistantReasoningMessage(content, reasoning, details),
		usage,
		reasoning,
		details,
		toolCalls...,
	)
}

// newUsage builds a genai usage from Messages API counters. The API reports
// input_tokens net of the cached prefix; genai's PromptTokens is the whole
// prompt, so the cache read and write counts are added back.
func newUsage(inputTokens, outputTokens, cacheReadTokens, cacheCreationTokens int64) llm.ChatCompletionUsage {
	promptTokens := inputTokens + cacheReadTokens + cacheCreationTokens
	return llm.NewChatCompletionUsageWithCacheCreation(
		promptTokens,
		outputTokens,
		promptTokens+outputTokens,
		cacheReadTokens,
		cacheCreationTokens,
	)
}

func excludeReasoning(opts *llm.ChatCompletionOptions) bool {
	return opts.Reasoning != nil && opts.Reasoning.Exclude
}

// mapError surfaces the upstream HTTP status through llm.HTTPError so the
// retry and rate limit machinery can act on it.
//
// An error event received in the middle of a stream (an overloaded_error,
// typically) inherits the 200 of the response that carried it; such an
// error is reported as a 503 so it stays retryable, and a rate limit error
// type is reported as a 429 whatever the transport status.
func mapError(err error) error {
	var apiErr *anthropicsdk.Error
	if !errors.As(err, &apiErr) {
		return err
	}

	body := apiErr.RawJSON()
	if body == "" {
		body = apiErr.Error()
	}

	status := apiErr.StatusCode
	switch {
	case apiErr.Type() == shared.ErrorTypeRateLimitError:
		status = http.StatusTooManyRequests
	case status < http.StatusBadRequest:
		status = http.StatusServiceUnavailable
	}

	return llm.RateLimitError(status, body)
}

func NewChatCompletionClient(client anthropicsdk.Client, model string, maxTokens int64) *ChatCompletionClient {
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}
	return &ChatCompletionClient{
		client:    client,
		model:     model,
		maxTokens: maxTokens,
	}
}

var _ llm.ChatCompletionClient = &ChatCompletionClient{}
var _ llm.ChatCompletionStreamingClient = &ChatCompletionClient{}

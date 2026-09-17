package anthropic

import (
	"context"
	"encoding/json"
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
	var accumulateErr error
	for stream.Next() {
		if err := message.Accumulate(stream.Current()); err != nil {
			accumulateErr = err
			break
		}
	}
	// The transport error, when there is one, is the actual cause and
	// carries the upstream status; an accumulation failure on its own is a
	// malformed stream, reported as a retryable upstream failure.
	if err := stream.Err(); err != nil {
		return nil, errors.WithStack(mapError(err))
	}
	if accumulateErr != nil {
		return nil, errors.WithStack(llm.RateLimitError(http.StatusBadGateway, "malformed message stream: "+accumulateErr.Error()))
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

// excludeReasoning reports whether the caller asked not to see the
// reasoning. Thinking stays enabled upstream and the signed blocks are still
// carried on the message: the next turn of an agent loop has to replay them
// or the API rejects it. Only the plaintext Reasoning() is withheld.
func excludeReasoning(opts *llm.ChatCompletionOptions) bool {
	return opts.Reasoning != nil && opts.Reasoning.Exclude
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

		emitter := newStreamEmitter(ctx, chunks, excludeReasoning(opts))

		for stream.Next() {
			emitter.handle(stream.Current())
		}
		if err := stream.Err(); err != nil {
			// The input tokens are known from message_start and the output
			// tokens from the last message_delta, so a stream that dies
			// mid-flight still reports what the provider billed. If nothing
			// was ever published the chunk carries no usage at all: a zeroed
			// usage would read as "this cost nothing" rather than "unknown".
			streamErr := errors.WithStack(mapError(err))
			if emitter.usageSeen {
				emitter.sendChunk(llm.NewErrorStreamChunkWithUsage(streamErr, emitter.usage()))
			} else {
				emitter.sendChunk(llm.NewErrorStreamChunk(streamErr))
			}
			return
		}

		// Same rule as ChatCompletion: a clean stream that opened no
		// content block at all is ErrNoMessage, not an empty assistant
		// turn. Counting blocks rather than deltas keeps both entry points
		// in step on a response made of an empty text block.
		if emitter.blocks == 0 {
			emitter.sendChunk(llm.NewErrorStreamChunk(errors.WithStack(llm.ErrNoMessage)))
			return
		}

		emitter.sendChunk(llm.NewCompleteStreamChunk(emitter.usage()))
	}()

	return chunks, nil
}

// streamEmitter translates Messages API events into genai stream chunks.
//
// Thinking text is streamed incrementally through Reasoning() for display,
// and the complete block (text plus signature) is emitted once as a
// ReasoningDetail when the block closes: a signature only verifies the exact
// text it was computed over, so a detail carrying one without the other
// cannot be replayed. Excluding the reasoning silences the incremental text
// only; the closing detail is always emitted so the turn can be replayed.
type streamEmitter struct {
	// ctx guards every send: a consumer that abandons the channel — a proxy
	// whose client hung up, an agent loop giving up — cancels it, and without
	// this the goroutine would block forever on an unread channel, leaking the
	// upstream HTTP response with it.
	ctx              context.Context
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
	// completeInput marks the tool_use blocks whose start event already
	// carried their whole input, so that input deltas repeating it are not
	// emitted twice. Text and thinking get no such shortcut: a gateway may
	// open a block with the beginning of its text only, and a thinking
	// block's signature always arrives as a delta.
	completeInput map[int64]bool
	// blocks counts the content blocks opened so far.
	blocks int

	inputTokens         int64
	outputTokens        int64
	cacheReadTokens     int64
	cacheCreationTokens int64
	// usageSeen marks that the provider published counters at least once, so
	// that deltas carry them instead of a zeroed usage.
	usageSeen bool
}

type thinkingBlock struct {
	text      string
	signature string
}

func newStreamEmitter(ctx context.Context, chunks chan<- llm.StreamChunk, excludeReasoning bool) *streamEmitter {
	return &streamEmitter{
		ctx:              ctx,
		chunks:           chunks,
		excludeReasoning: excludeReasoning,
		toolIndexes:      map[int64]int{},
		thinking:         map[int64]*thinkingBlock{},
		completeInput:    map[int64]bool{},
	}
}

// send emits one delta chunk, carrying the usage known so far once the
// provider has published any.
func (e *streamEmitter) send(delta llm.StreamDelta) {
	if e.usageSeen {
		e.sendChunk(llm.NewStreamChunkWithUsage(delta, e.usage()))
		return
	}
	e.sendChunk(llm.NewStreamChunk(delta))
}

// sendChunk writes one chunk to the channel, giving up if the consumer is gone.
func (e *streamEmitter) sendChunk(chunk llm.StreamChunk) {
	select {
	case e.chunks <- chunk:
	case <-e.ctx.Done():
	}
}

// emitDetail sends one complete reasoning detail, numbered in emission order.
func (e *streamEmitter) emitDetail(detail llm.ReasoningDetail) {
	detail.Index = e.detailCount
	e.detailCount++
	e.send(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", "", []llm.ReasoningDetail{detail}))
}

func (e *streamEmitter) handle(event anthropicsdk.MessageStreamEventUnion) {
	switch event.Type {
	case "message_start":
		e.recordUsage(event.Message.Usage)

	case "message_delta":
		if event.Usage.JSON.OutputTokens.Valid() {
			e.outputTokens = event.Usage.OutputTokens
			e.usageSeen = true
		}
		if event.Usage.JSON.InputTokens.Valid() {
			e.inputTokens = event.Usage.InputTokens
			e.usageSeen = true
		}
		if event.Usage.JSON.CacheReadInputTokens.Valid() {
			e.cacheReadTokens = event.Usage.CacheReadInputTokens
		}
		if event.Usage.JSON.CacheCreationInputTokens.Valid() {
			e.cacheCreationTokens = event.Usage.CacheCreationInputTokens
		}

	case "content_block_start":
		e.blocks++
		block := event.ContentBlock
		switch block.Type {
		case "text":
			if block.Text != "" {
				e.send(llm.NewStreamDelta(llm.RoleAssistant, block.Text))
			}
		case "tool_use":
			index := e.toolCount
			e.toolCount++
			e.toolIndexes[event.Index] = index
			// The official API always streams the arguments as
			// input_json_delta events and opens the block with an empty
			// object; some gateways ship the whole input here instead.
			initial := initialToolInput(block.Input)
			if initial != "" {
				e.completeInput[event.Index] = true
			}
			e.send(llm.NewStreamDelta(llm.RoleAssistant, "",
				llm.NewToolCallDelta(index, block.ID, block.Name, initial),
			))
		case "thinking":
			e.thinking[event.Index] = &thinkingBlock{text: block.Thinking, signature: block.Signature}
			// Same courtesy as for tool inputs: a gateway may open the
			// block with its whole text.
			if block.Thinking != "" && !e.excludeReasoning {
				e.send(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", block.Thinking, nil))
			}
		case "redacted_thinking":
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
		e.emitDetail(llm.ReasoningDetail{
			Type:      llm.ReasoningDetailTypeText,
			Text:      block.text,
			Signature: block.signature,
		})

	case "content_block_delta":
		delta := event.Delta
		switch delta.Type {
		case "text_delta":
			e.send(llm.NewStreamDelta(llm.RoleAssistant, delta.Text))
		case "input_json_delta":
			index, known := e.toolIndexes[event.Index]
			if !known || delta.PartialJSON == "" || e.completeInput[event.Index] {
				// A gateway that shipped the whole input on the start
				// event and streams it again would double the JSON.
				return
			}
			e.send(llm.NewStreamDelta(llm.RoleAssistant, "",
				llm.NewToolCallDelta(index, "", "", delta.PartialJSON),
			))
		case "thinking_delta":
			if block, open := e.thinking[event.Index]; open {
				block.text += delta.Thinking
			}
			if e.excludeReasoning || delta.Thinking == "" {
				return
			}
			e.send(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", delta.Thinking, nil))
		case "signature_delta":
			if block, open := e.thinking[event.Index]; open {
				block.signature += delta.Signature
			}
		}
	}
}

func (e *streamEmitter) recordUsage(usage anthropicsdk.Usage) {
	e.usageSeen = true
	e.inputTokens = usage.InputTokens
	e.outputTokens = usage.OutputTokens
	e.cacheReadTokens = usage.CacheReadInputTokens
	e.cacheCreationTokens = usage.CacheCreationInputTokens
}

func (e *streamEmitter) usage() llm.ChatCompletionUsage {
	return newUsage(e.inputTokens, e.outputTokens, e.cacheReadTokens, e.cacheCreationTokens)
}

// initialToolInput renders the tool input carried by a content_block_start
// event, or "" when it is the usual empty placeholder.
func initialToolInput(input any) string {
	if input == nil {
		return ""
	}
	if m, ok := input.(map[string]any); ok && len(m) == 0 {
		return ""
	}
	raw, err := json.Marshal(input)
	if err != nil {
		return ""
	}
	return string(raw)
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

	if reasoning == "" && len(details) == 0 {
		return llm.NewChatCompletionResponse(llm.NewMessage(llm.RoleAssistant, content), usage, toolCalls...)
	}

	// The signed blocks stay on the message whatever the caller asked: they
	// are what the next turn replays. Excluding only hides the plaintext.
	if excludeReasoning {
		reasoning = ""
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

// mapError surfaces the upstream HTTP status through llm.HTTPError so the
// retry and rate limit machinery can act on it. llm.RateLimitError is the
// generic constructor for an upstream failure despite its name: it only
// tags a 429 as a rate limit.
//
// An error event received in the middle of a stream inherits the 200 of the
// response that carried it. The status is then recovered from the error
// type, so an overloaded upstream stays retryable while a rejected request
// is not replayed to the same outcome.
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
	if status < http.StatusBadRequest {
		status = statusForErrorType(apiErr.Type())
	}

	return llm.RateLimitError(status, body)
}

// statusForErrorType maps an Anthropic error type to the HTTP status the API
// documents for it.
func statusForErrorType(errorType shared.ErrorType) int {
	switch errorType {
	case shared.ErrorTypeInvalidRequestError:
		return http.StatusBadRequest
	case shared.ErrorTypeAuthenticationError:
		return http.StatusUnauthorized
	case shared.ErrorTypeBillingError:
		return http.StatusPaymentRequired
	case shared.ErrorTypePermissionError:
		return http.StatusForbidden
	case shared.ErrorTypeNotFoundError:
		return http.StatusNotFound
	case shared.ErrorTypeTimeoutError:
		// Transient, like an overload: a gateway status, which stays
		// retryable, rather than the client-side 408.
		return http.StatusGatewayTimeout
	case shared.ErrorTypeRateLimitError:
		return http.StatusTooManyRequests
	case shared.ErrorTypeAPIError:
		return http.StatusInternalServerError
	default:
		// overloaded_error and anything new: transient until proven otherwise.
		return http.StatusServiceUnavailable
	}
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

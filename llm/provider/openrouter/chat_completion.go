package openrouter

import (
	"math"
	"slices"
	"sync/atomic"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/context"
	"github.com/pkg/errors"
	"github.com/revrost/go-openrouter"
)

type ChatCompletionClient struct {
	client *openrouter.Client
	model  string
}

// toOpenRouterReasoning maps llm.ReasoningOptions to openrouter.ChatCompletionReasoning.
func toOpenRouterReasoning(opts *llm.ReasoningOptions) *openrouter.ChatCompletionReasoning {
	if opts == nil {
		return nil
	}
	r := &openrouter.ChatCompletionReasoning{}
	if opts.Effort != nil {
		effort := string(*opts.Effort)
		r.Effort = &effort
	}
	if opts.MaxTokens != nil {
		r.MaxTokens = opts.MaxTokens
	}
	if opts.Exclude {
		exclude := true
		r.Exclude = &exclude
	}
	if opts.Enabled != nil {
		r.Enabled = opts.Enabled
	}
	return r
}

// toReasoningDetails maps a slice of openrouter.ChatCompletionReasoningDetails to []llm.ReasoningDetail.
func toReasoningDetails(details []openrouter.ChatCompletionReasoningDetails) []llm.ReasoningDetail {
	if len(details) == 0 {
		return nil
	}
	result := make([]llm.ReasoningDetail, 0, len(details))
	for _, d := range details {
		result = append(result, llm.ReasoningDetail{
			ID:      d.ID,
			Type:    llm.ReasoningDetailType(d.Type),
			Text:    d.Text,
			Summary: d.Summary,
			Data:    d.Data,
			Format:  d.Format,
			Index:   d.Index,
		})
	}
	return result
}

// fromReasoningDetails maps a slice of llm.ReasoningDetail to []openrouter.ChatCompletionReasoningDetails.
func fromReasoningDetails(details []llm.ReasoningDetail) []openrouter.ChatCompletionReasoningDetails {
	if len(details) == 0 {
		return nil
	}
	result := make([]openrouter.ChatCompletionReasoningDetails, 0, len(details))
	for _, d := range details {
		result = append(result, openrouter.ChatCompletionReasoningDetails{
			ID:      d.ID,
			Type:    openrouter.ChatCompletionReasoningDetailsType(d.Type),
			Text:    d.Text,
			Summary: d.Summary,
			Data:    d.Data,
			Format:  d.Format,
			Index:   d.Index,
		})
	}
	return result
}

// toOpenRouterCacheControl converts an llm.CacheControl hint to its OpenRouter
// SDK equivalent, forwarded as-is to the provider.
func toOpenRouterCacheControl(cc *llm.CacheControl) *openrouter.CacheControl {
	if cc == nil {
		return nil
	}
	return &openrouter.CacheControl{
		Type: cc.Type,
		TTL:  cc.TTL,
	}
}

// messageCacheControl returns the cache hint carried by m, if any.
func messageCacheControl(m llm.Message) *llm.CacheControl {
	if cm, ok := m.(llm.CacheControlMessage); ok {
		return cm.CacheControl()
	}
	return nil
}

// withCacheControl puts cc on the last part of content, switching a plain
// text content to its multi-part form since only parts carry the hint. A
// marked empty text block is refused upstream, so trailing empty text parts,
// as an empty document gives, are passed over; it reports false when no part
// is left to carry the hint.
func withCacheControl(content openrouter.Content, cc *llm.CacheControl) (openrouter.Content, bool) {
	if cc == nil {
		return content, true
	}
	if len(content.Multi) == 0 {
		if content.Text == "" {
			return content, false
		}
		content = openrouter.Content{
			Multi: []openrouter.ChatMessagePart{{
				Type: openrouter.ChatMessagePartTypeText,
				Text: content.Text,
			}},
		}
	}
	for i := len(content.Multi) - 1; i >= 0; i-- {
		if part := content.Multi[i]; part.Type == openrouter.ChatMessagePartTypeText && part.Text == "" {
			continue
		}
		parts := slices.Clone(content.Multi)
		parts[i].CacheControl = toOpenRouterCacheControl(cc)
		return openrouter.Content{Multi: parts}, true
	}
	return content, false
}

// messageContent builds the content of a user or tool message: its text
// alone, or its text followed by one or more parts per attachment, each
// converted by ConvertAttachmentToContent whatever their number.
func messageContent(m llm.Message) (openrouter.Content, error) {
	if len(m.Attachments()) == 0 {
		return openrouter.Content{Text: m.Content()}, nil
	}

	parts := make([]openrouter.ChatMessagePart, 0, len(m.Attachments())+1)

	if m.Content() != "" {
		parts = append(parts, openrouter.ChatMessagePart{
			Type: openrouter.ChatMessagePartTypeText,
			Text: m.Content(),
		})
	}

	for i, attachment := range m.Attachments() {
		// The message text is added once above, not before each attachment.
		content, err := ConvertAttachmentToContent(attachment, "")
		if err != nil {
			return openrouter.Content{}, errors.Wrapf(err, "failed to convert attachment %d to content", i)
		}
		// An empty document gives an empty text part, which the upstream may
		// refuse: it is left out.
		for _, part := range content.Multi {
			if part.Type == openrouter.ChatMessagePartTypeText && part.Text == "" {
				continue
			}
			parts = append(parts, part)
		}
	}

	if len(parts) == 0 {
		// Nothing left, as for a text-only message with no text: the
		// content is omitted rather than sent as null.
		return openrouter.Content{}, nil
	}

	return openrouter.Content{Multi: parts}, nil
}

// buildMessages converts llm.Message slice to openrouter.ChatCompletionMessage slice,
// handling attachments and reasoning preservation.
func buildMessages(msgs []llm.Message, model string) ([]openrouter.ChatCompletionMessage, error) {
	messages := make([]openrouter.ChatCompletionMessage, 0, len(msgs))

	// Create validator for provider-specific validation (Layer 2)
	validator := NewOpenRouterAttachmentValidator(model)

	for _, m := range msgs {
		// Validate attachments (Layer 2 - provider-specific validation)
		for _, attachment := range m.Attachments() {
			if err := validator.ValidateAttachment(attachment); err != nil {
				return nil, errors.Wrapf(err, "attachment validation failed for message with role %s", m.Role())
			}
		}

		var message openrouter.ChatCompletionMessage

		switch m.Role() {
		case llm.RoleSystem:
			if len(m.Attachments()) > 0 {
				return nil, errors.Errorf("system messages cannot have attachments")
			}
			message = openrouter.ChatCompletionMessage{
				Role:    openrouter.ChatMessageRoleSystem,
				Content: openrouter.Content{Text: m.Content()},
			}
		case llm.RoleUser:
			content, err := messageContent(m)
			if err != nil {
				return nil, errors.WithStack(err)
			}
			message = openrouter.ChatCompletionMessage{
				Role:    openrouter.ChatMessageRoleUser,
				Content: content,
			}
		case llm.RoleAssistant:
			if len(m.Attachments()) > 0 {
				return nil, errors.Errorf("assistant messages cannot have attachments")
			}
			message = openrouter.ChatCompletionMessage{
				Role:    openrouter.ChatMessageRoleAssistant,
				Content: openrouter.Content{Text: m.Content()},
			}
			// Preserve reasoning for multi-turn conversations.
			// When a model returns reasoning tokens, they must be passed back in
			// subsequent requests so the model can continue its reasoning chain.
			preserveReasoning(&message, m)
		case llm.RoleTool:
			toolMessage, ok := m.(llm.ToolMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool message type '%T'", m)
			}
			content, err := messageContent(m)
			if err != nil {
				return nil, errors.WithStack(err)
			}
			message = openrouter.ChatCompletionMessage{
				Role:       openrouter.ChatMessageRoleTool,
				ToolCallID: toolMessage.ID(),
				Content:    content,
			}
		case llm.RoleToolCalls:
			if len(m.Attachments()) > 0 {
				return nil, errors.Errorf("tool calls messages cannot have attachments")
			}
			toolCallsMessage, ok := m.(llm.ToolCallsMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool calls message type '%T'", m)
			}

			// The text the model wrote alongside its tool calls is part of
			// the turn, and the only part a cache hint can land on.
			message = openrouter.ChatCompletionMessage{
				Role:    openrouter.ChatMessageRoleAssistant,
				Content: openrouter.Content{Text: m.Content()},
			}

			toolCalls := make([]openrouter.ToolCall, 0, len(toolCallsMessage.ToolCalls()))
			for _, tc := range toolCallsMessage.ToolCalls() {
				arguments, ok := tc.Parameters().(string)
				if !ok {
					return nil, errors.Errorf("expected string parameters for tool call %s, got %T", tc.ID(), tc.Parameters())
				}

				toolCalls = append(toolCalls, openrouter.ToolCall{
					ID: tc.ID(),
					Function: openrouter.FunctionCall{
						Name:      tc.Name(),
						Arguments: arguments,
					},
					Type: openrouter.ToolTypeFunction,
				})
			}
			message.ToolCalls = toolCalls

			// Preserve reasoning alongside tool calls.
			// For reasoning models (e.g. Claude, GPT-5), when the model responds with
			// tool calls AND reasoning, both must be sent back together in the next turn
			// so the model can continue its reasoning chain from where it left off.
			preserveReasoning(&message, m)
		default:
			continue
		}

		// A hint means "cache everything so far". When the message has no
		// part to carry it, as a tool calls message without text, it goes
		// on the end of the closest earlier message that has one: unlike
		// the anthropic provider, the previous message may itself be tool
		// calls without text, whose calls cannot carry a hint here. A hint
		// already there is replaced, as the anthropic provider does when
		// two land on the same block. With no such message nothing precedes
		// it and the hint is void.
		cc := messageCacheControl(m)
		var placed bool
		message.Content, placed = withCacheControl(message.Content, cc)
		if !placed {
			for i := len(messages) - 1; i >= 0 && !placed; i-- {
				messages[i].Content, placed = withCacheControl(messages[i].Content, cc)
			}
		}

		messages = append(messages, message)
	}

	return messages, nil
}

// preserveReasoning copies the reasoning carried by m, if any, onto message.
func preserveReasoning(message *openrouter.ChatCompletionMessage, m llm.Message) {
	rm, ok := m.(llm.ReasoningMessage)
	if !ok {
		return
	}
	if r := rm.Reasoning(); r != "" {
		message.Reasoning = &r
	}
	if details := rm.ReasoningDetails(); len(details) > 0 {
		message.ReasoningDetails = fromReasoningDetails(details)
	}
}

// ChatCompletion implements llm.Client.
func (c *ChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	// Validate options before proceeding
	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	req := openrouter.ChatCompletionRequest{
		Model: c.model,
		Usage: &openrouter.IncludeUsage{Include: true},
	}
	if opts.Temperature != nil {
		req.Temperature = float32(*opts.Temperature)
	}

	// Configure reasoning if requested
	if opts.Reasoning != nil {
		req.Reasoning = toOpenRouterReasoning(opts.Reasoning)
	}

	if opts.ResponseFormat == llm.ResponseFormatJSON {
		jsonFormat := openrouter.ChatCompletionResponseFormat{
			Type: openrouter.ChatCompletionResponseFormatTypeJSONSchema,
		}

		if opts.ResponseSchema != nil {
			jsonFormat.JSONSchema = &openrouter.ChatCompletionResponseFormatJSONSchema{
				Name:        opts.ResponseSchema.Name(),
				Description: opts.ResponseSchema.Description(),
				Schema:      jsonMarshaller{opts.ResponseSchema.Schema()},
				Strict:      llm.IsStrictResponseSchema(opts.ResponseSchema),
			}
		}

		req.ResponseFormat = &jsonFormat
	}

	if len(opts.Tools) > 0 {
		tools := make([]openrouter.Tool, 0, len(opts.Tools))

		for _, t := range opts.Tools {
			tools = append(tools, openrouter.Tool{
				Type: openrouter.ToolTypeFunction,
				Function: &openrouter.FunctionDefinition{
					Name:        t.Name(),
					Description: t.Description(),
					Parameters:  t.Parameters(),
				},
			})
		}

		req.Tools = tools
	}

	if opts.ToolChoice != "" {
		req.ToolChoice = string(opts.ToolChoice)
	}

	// Configure modalities (e.g., ["text", "audio"] for audio output)
	if len(opts.Modalities) > 0 {
		modalities := make([]openrouter.ChatCompletionModality, len(opts.Modalities))
		for i, m := range opts.Modalities {
			modalities[i] = openrouter.ChatCompletionModality(m)
		}
		req.Modalities = modalities
	}

	// Configure audio output
	if opts.Audio != nil {
		req.AudioConfig = &openrouter.ChatCompletionAudioConfig{
			Voice:  openrouter.AudioVoice(opts.Audio.Voice),
			Format: openrouter.AudioFormat(opts.Audio.Format),
		}
	}

	messages, err := buildMessages(opts.Messages, c.model)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	req.Messages = messages

	if opts.Seed != nil {
		req.Seed = opts.Seed
	}

	if opts.MaxCompletionTokens != nil {
		req.MaxCompletionTokens = *opts.MaxCompletionTokens
	}

	transforms, err := ContextTransforms(ctx)
	if err != nil && !errors.Is(err, context.ErrNotFound) {
		return nil, errors.WithStack(err)
	}

	req.Transforms = transforms

	models, err := ContextModels(ctx)
	if err != nil && !errors.Is(err, context.ErrNotFound) {
		return nil, errors.WithStack(err)
	}

	req.Models = models

	if opts.SessionID != "" {
		req.SessionId = opts.SessionID
	}

	res, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		var reqErr *openrouter.RequestError
		if errors.As(err, &reqErr) {
			return nil, errors.WithStack(llm.RateLimitError(reqErr.HTTPStatusCode, reqErr.Error()))
		}

		return nil, errors.WithStack(err)
	}

	if len(res.Choices) == 0 {
		return nil, errors.WithStack(llm.ErrNoMessage)
	}

	openrouterMessage := res.Choices[0].Message

	// Extract reasoning from the response message.
	// Prefer the structured reasoning_details when available (supports encrypted blocks),
	// fall back to the plain reasoning string.
	var (
		reasoning        string
		reasoningDetails []llm.ReasoningDetail
	)

	if openrouterMessage.Reasoning != nil {
		reasoning = *openrouterMessage.Reasoning
	} else if openrouterMessage.ReasoningContent != nil {
		reasoning = *openrouterMessage.ReasoningContent
	}

	if len(openrouterMessage.ReasoningDetails) > 0 {
		reasoningDetails = toReasoningDetails(openrouterMessage.ReasoningDetails)
	}

	// Build the response message. When reasoning is present, return a ReasoningMessage
	// so callers can preserve it across turns.
	var message llm.Message
	if reasoning != "" || len(reasoningDetails) > 0 {
		message = llm.NewAssistantReasoningMessage(openrouterMessage.Content.Text, reasoning, reasoningDetails)
	} else {
		message = llm.NewMessage(llm.RoleAssistant, openrouterMessage.Content.Text)
	}

	toolCalls := make([]llm.ToolCall, 0)

	for _, tc := range openrouterMessage.ToolCalls {
		toolCalls = append(toolCalls, llm.NewToolCall(tc.ID, tc.Function.Name, tc.Function.Arguments))
	}

	usage := llm.NewChatCompletionUsageWithCost(
		int64(res.Usage.PromptTokens),
		int64(res.Usage.CompletionTokens),
		int64(res.Usage.TotalTokens),
		int64(res.Usage.PromptTokenDetails.CachedTokens),
		res.Usage.Cost,
		"USD", // OpenRouter always reports cost in USD
	)

	return llm.NewChatCompletionResponseWithReasoning(message, usage, reasoning, reasoningDetails, toolCalls...), nil
}

// ChatCompletionStream implements llm.ChatCompletionStreamingClient.
func (c *ChatCompletionClient) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	// Validate options before proceeding
	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	req := openrouter.ChatCompletionRequest{
		Model:         c.model,
		Stream:        true, // Enable streaming
		StreamOptions: &openrouter.StreamOptions{IncludeUsage: true},
		Usage:         &openrouter.IncludeUsage{Include: true},
	}
	if opts.Temperature != nil {
		req.Temperature = float32(*opts.Temperature)
	}

	// Configure reasoning if requested
	if opts.Reasoning != nil {
		req.Reasoning = toOpenRouterReasoning(opts.Reasoning)
	}

	if opts.ResponseFormat == llm.ResponseFormatJSON {
		jsonFormat := openrouter.ChatCompletionResponseFormat{
			Type: openrouter.ChatCompletionResponseFormatTypeJSONSchema,
		}

		if opts.ResponseSchema != nil {
			jsonFormat.JSONSchema = &openrouter.ChatCompletionResponseFormatJSONSchema{
				Name:        opts.ResponseSchema.Name(),
				Description: opts.ResponseSchema.Description(),
				Schema:      jsonMarshaller{opts.ResponseSchema.Schema()},
				Strict:      llm.IsStrictResponseSchema(opts.ResponseSchema),
			}
		}

		req.ResponseFormat = &jsonFormat
	}

	if len(opts.Tools) > 0 {
		tools := make([]openrouter.Tool, 0, len(opts.Tools))

		for _, t := range opts.Tools {
			tools = append(tools, openrouter.Tool{
				Type: openrouter.ToolTypeFunction,
				Function: &openrouter.FunctionDefinition{
					Name:        t.Name(),
					Description: t.Description(),
					Parameters:  t.Parameters(),
				},
			})
		}

		req.Tools = tools
	}

	if opts.ToolChoice != "" {
		req.ToolChoice = string(opts.ToolChoice)
	}

	// Configure modalities (e.g., ["text", "audio"] for audio output)
	if len(opts.Modalities) > 0 {
		modalities := make([]openrouter.ChatCompletionModality, len(opts.Modalities))
		for i, m := range opts.Modalities {
			modalities[i] = openrouter.ChatCompletionModality(m)
		}
		req.Modalities = modalities
	}

	// Configure audio output
	if opts.Audio != nil {
		req.AudioConfig = &openrouter.ChatCompletionAudioConfig{
			Voice:  openrouter.AudioVoice(opts.Audio.Voice),
			Format: openrouter.AudioFormat(opts.Audio.Format),
		}
	}

	messages, err := buildMessages(opts.Messages, c.model)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	req.Messages = messages

	if opts.Seed != nil {
		req.Seed = opts.Seed
	}

	if opts.MaxCompletionTokens != nil {
		req.MaxCompletionTokens = *opts.MaxCompletionTokens
	}

	transforms, err := ContextTransforms(ctx)
	if err != nil && !errors.Is(err, context.ErrNotFound) {
		return nil, errors.WithStack(err)
	}

	req.Transforms = transforms

	models, err := ContextModels(ctx)
	if err != nil && !errors.Is(err, context.ErrNotFound) {
		return nil, errors.WithStack(err)
	}

	req.Models = models

	if opts.SessionID != "" {
		req.SessionId = opts.SessionID
	}

	// Create streaming channel
	chunks := make(chan llm.StreamChunk, 10)

	var (
		promptTokens     atomic.Int64
		completionTokens atomic.Int64
		totalTokens      atomic.Int64
		cachedTokens     atomic.Int64
		cost             atomic.Uint64 // float64 bits, see math.Float64bits/Float64frombits
		costReported     atomic.Bool
		usageReported    atomic.Bool
	)

	// send gives up if the consumer abandoned the channel; sendTerminal takes
	// the buffer slot first so that the chunk ending the stream survives a
	// cancellation it is often there to report. See llm.SendChunk.
	send := func(chunk llm.StreamChunk) { llm.SendChunk(ctx, chunks, chunk) }
	sendTerminal := func(chunk llm.StreamChunk) { llm.SendTerminalChunk(ctx, chunks, chunk) }

	// currentUsage snapshots the counters the gateway has published so far.
	// OpenRouter is asked for usage explicitly, and sends it as the stream goes,
	// so carrying it on the deltas keeps it available to a consumer whose stream
	// is cut short before the final chunk.
	currentUsage := func() llm.ChatCompletionUsage {
		if costReported.Load() {
			return llm.NewChatCompletionUsageWithCost(
				promptTokens.Load(),
				completionTokens.Load(),
				totalTokens.Load(),
				cachedTokens.Load(),
				math.Float64frombits(cost.Load()),
				"USD", // OpenRouter always reports cost in USD
			)
		}
		return llm.NewChatCompletionUsageWithCache(
			promptTokens.Load(),
			completionTokens.Load(),
			totalTokens.Load(),
			cachedTokens.Load(),
		)
	}

	// sendDelta carries the running counters when the gateway has published any.
	sendDelta := func(delta llm.StreamDelta) {
		if usageReported.Load() {
			send(llm.NewStreamChunkWithUsage(delta, currentUsage()))
			return
		}
		send(llm.NewStreamChunk(delta))
	}

	// sendStreamError carries them too: the provider billed what it produced
	// before it failed.
	sendStreamError := func(err error) {
		if usageReported.Load() {
			sendTerminal(llm.NewErrorStreamChunkWithUsage(err, currentUsage()))
			return
		}
		sendTerminal(llm.NewErrorStreamChunk(err))
	}

	go func() {
		defer close(chunks)

		stream, err := c.client.CreateChatCompletionStream(ctx, req)
		if err != nil {
			var reqErr *openrouter.RequestError
			if errors.As(err, &reqErr) {
				sendStreamError(errors.WithStack(llm.RateLimitError(reqErr.HTTPStatusCode, reqErr.Error())))
				return
			}
			sendStreamError(errors.WithStack(err))
			return
		}
		defer stream.Close()

		for {
			response, err := stream.Recv()
			if err != nil {
				if err.Error() == "EOF" {
					// Stream ended normally
					break
				}
				sendStreamError(errors.WithStack(err))
				return
			}

			if response.Usage != nil {
				promptTokens.Store(int64(response.Usage.PromptTokens))
				completionTokens.Store(int64(response.Usage.CompletionTokens))
				totalTokens.Store(int64(response.Usage.TotalTokens))
				cachedTokens.Store(int64(response.Usage.PromptTokenDetails.CachedTokens))
				cost.Store(math.Float64bits(response.Usage.Cost))
				costReported.Store(true)
				usageReported.Store(true)
			}

			if len(response.Choices) == 0 {
				continue
			}

			choice := response.Choices[0]
			delta := choice.Delta

			// Create stream delta
			var toolCallDeltas []llm.ToolCallDelta
			for _, tc := range delta.ToolCalls {
				idx := 0
				if tc.Index != nil {
					idx = *tc.Index
				}
				toolCallDeltas = append(toolCallDeltas, llm.NewToolCallDelta(
					idx,
					tc.ID,
					tc.Function.Name,
					tc.Function.Arguments,
				))
			}

			// Extract incremental reasoning from the delta.
			// When present, emit a ReasoningStreamDelta so callers can accumulate
			// reasoning tokens for display or preservation.
			var (
				deltaReasoning        string
				deltaReasoningDetails []llm.ReasoningDetail
			)

			if delta.Reasoning != nil {
				deltaReasoning = *delta.Reasoning
			} else if delta.ReasoningContent != "" {
				deltaReasoning = delta.ReasoningContent
			}

			if len(delta.ReasoningDetails) > 0 {
				deltaReasoningDetails = toReasoningDetails(delta.ReasoningDetails)
			}

			// Extract audio data from the delta (for text-to-audio models like Lyria)
			var audioData, transcript string
			if delta.Audio != nil {
				audioData = delta.Audio.Data
				transcript = delta.Audio.Transcript
			}

			// Emit audio stream delta if audio data or transcript is present
			if audioData != "" || transcript != "" {
				streamDelta := llm.NewAudioStreamDelta(
					llm.RoleAssistant,
					delta.Content,
					audioData,
					transcript,
					toolCallDeltas...,
				)
				sendDelta(streamDelta)
			} else if deltaReasoning != "" || len(deltaReasoningDetails) > 0 {
				streamDelta := llm.NewReasoningStreamDelta(
					llm.RoleAssistant,
					delta.Content,
					deltaReasoning,
					deltaReasoningDetails,
					toolCallDeltas...,
				)
				sendDelta(streamDelta)
			} else {
				streamDelta := llm.NewStreamDelta(
					llm.RoleAssistant,
					delta.Content,
					toolCallDeltas...,
				)
				sendDelta(streamDelta)
			}
		}

		sendTerminal(llm.NewCompleteStreamChunk(currentUsage()))
	}()

	return chunks, nil
}

func NewChatCompletionClient(client *openrouter.Client, model string) *ChatCompletionClient {
	return &ChatCompletionClient{
		client: client,
		model:  model,
	}
}

var _ llm.ChatCompletionClient = &ChatCompletionClient{}
var _ llm.ChatCompletionStreamingClient = &ChatCompletionClient{}

package openrouter

import (
	"fmt"
	"strings"
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

		switch m.Role() {
		case llm.RoleSystem:
			if len(m.Attachments()) > 0 {
				return nil, errors.Errorf("system messages cannot have attachments")
			}
			messages = append(messages, openrouter.ChatCompletionMessage{
				Role: openrouter.ChatMessageRoleSystem,
				Content: openrouter.Content{
					Text: m.Content(),
				},
			})
		case llm.RoleUser:
			if len(m.Attachments()) > 0 {
				// Handle multimodal user message
				if len(m.Attachments()) == 1 {
					// Single attachment - use the conversion function
					content, err := ConvertAttachmentToContent(m.Attachments()[0], m.Content())
					if err != nil {
						return nil, errors.Wrapf(err, "failed to convert attachment to content")
					}
					messages = append(messages, openrouter.ChatCompletionMessage{
						Role:    openrouter.ChatMessageRoleUser,
						Content: content,
					})
				} else {
					// Multiple attachments - build multi-part content manually
					parts := make([]openrouter.ChatMessagePart, 0)

					// Add text content if present
					if m.Content() != "" {
						parts = append(parts, openrouter.ChatMessagePart{
							Type: openrouter.ChatMessagePartTypeText,
							Text: m.Content(),
						})
					}

					// Add all attachments
					for _, attachment := range m.Attachments() {
						switch attachment.Type() {
						case llm.AttachmentTypeImage:
							data := attachment.Data()
							if attachment.Source() == llm.AttachmentSourceBase64 && !strings.HasPrefix(data, "data:") {
								data = fmt.Sprintf("data:%s;base64,%s", attachment.MimeType(), data)
							}
							parts = append(parts, openrouter.ChatMessagePart{
								Type: openrouter.ChatMessagePartTypeImageURL,
								ImageURL: &openrouter.ChatMessageImageURL{
									URL: data,
								},
							})
						default:
							return nil, errors.Errorf("unsupported attachment type: %s", attachment.Type())
						}
					}

					messages = append(messages, openrouter.ChatCompletionMessage{
						Role: openrouter.ChatMessageRoleUser,
						Content: openrouter.Content{
							Multi: parts,
						},
					})
				}
			} else {
				messages = append(messages, openrouter.ChatCompletionMessage{
					Role: openrouter.ChatMessageRoleUser,
					Content: openrouter.Content{
						Text: m.Content(),
					},
				})
			}
		case llm.RoleAssistant:
			if len(m.Attachments()) > 0 {
				return nil, errors.Errorf("assistant messages cannot have attachments")
			}
			msg := openrouter.ChatCompletionMessage{
				Role: openrouter.ChatMessageRoleAssistant,
				Content: openrouter.Content{
					Text: m.Content(),
				},
			}
			// Preserve reasoning for multi-turn conversations.
			// When a model returns reasoning tokens, they must be passed back in
			// subsequent requests so the model can continue its reasoning chain.
			if rm, ok := m.(llm.ReasoningMessage); ok {
				if r := rm.Reasoning(); r != "" {
					msg.Reasoning = &r
				}
				if details := rm.ReasoningDetails(); len(details) > 0 {
					msg.ReasoningDetails = fromReasoningDetails(details)
				}
			}
			messages = append(messages, msg)
		case llm.RoleTool:
			toolMessage, ok := m.(llm.ToolMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool message type '%T'", m)
			}

			if len(m.Attachments()) > 0 {
				// Handle multimodal user message
				if len(m.Attachments()) == 1 {
					// Single attachment - use the conversion function
					content, err := ConvertAttachmentToContent(m.Attachments()[0], m.Content())
					if err != nil {
						return nil, errors.Wrapf(err, "failed to convert attachment to content")
					}
					messages = append(messages, openrouter.ChatCompletionMessage{
						Role:       openrouter.ChatMessageRoleTool,
						ToolCallID: toolMessage.ID(),
						Content:    content,
					})
				} else {
					// Multiple attachments - build multi-part content manually
					parts := make([]openrouter.ChatMessagePart, 0)

					// Add text content if present
					if m.Content() != "" {
						parts = append(parts, openrouter.ChatMessagePart{
							Type: openrouter.ChatMessagePartTypeText,
							Text: m.Content(),
						})
					}

					// Add all attachments
					for _, attachment := range m.Attachments() {
						switch attachment.Type() {
						case llm.AttachmentTypeImage:
							data := attachment.Data()
							if attachment.Source() == llm.AttachmentSourceBase64 && !strings.HasPrefix(data, "data:") {
								data = fmt.Sprintf("data:%s;base64,%s", attachment.MimeType(), data)
							}
							parts = append(parts, openrouter.ChatMessagePart{
								Type: openrouter.ChatMessagePartTypeImageURL,
								ImageURL: &openrouter.ChatMessageImageURL{
									URL: data,
								},
							})
						default:
							return nil, errors.Errorf("unsupported attachment type: %s", attachment.Type())
						}
					}

					messages = append(messages, openrouter.ChatCompletionMessage{
						Role:       openrouter.ChatMessageRoleTool,
						ToolCallID: toolMessage.ID(),
						Content: openrouter.Content{
							Multi: parts,
						},
					})
				}
			} else {
				messages = append(messages, openrouter.ChatCompletionMessage{
					Role:       openrouter.ChatMessageRoleTool,
					ToolCallID: toolMessage.ID(),
					Content: openrouter.Content{
						Text: m.Content(),
					},
				})
			}
		case llm.RoleToolCalls:
			if len(m.Attachments()) > 0 {
				return nil, errors.Errorf("tool calls messages cannot have attachments")
			}
			toolCallsMessage, ok := m.(llm.ToolCallsMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool calls message type '%T'", m)
			}

			message := openrouter.ChatCompletionMessage{
				Role: openrouter.ChatMessageRoleAssistant,
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
			if rm, ok := m.(llm.ReasoningMessage); ok {
				if r := rm.Reasoning(); r != "" {
					message.Reasoning = &r
				}
				if details := rm.ReasoningDetails(); len(details) > 0 {
					message.ReasoningDetails = fromReasoningDetails(details)
				}
			}

			messages = append(messages, message)
		}
	}

	return messages, nil
}

// ChatCompletion implements llm.Client.
func (c *ChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	// Validate options before proceeding
	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	temperature := float32(opts.Temperature)

	req := openrouter.ChatCompletionRequest{
		Model:       c.model,
		Temperature: temperature,
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
				Strict:      true,
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

	req.ToolChoice = opts.ToolChoice

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

	res, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		var reqErr *openrouter.RequestError
		if errors.As(err, &reqErr) {
			return nil, errors.WithStack(llm.NewHTTPError(reqErr.HTTPStatusCode, reqErr.Error()))
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

	usage := llm.NewChatCompletionUsage(int64(res.Usage.PromptTokens), int64(res.Usage.CompletionTokens), int64(res.Usage.TotalTokens))

	return llm.NewChatCompletionResponseWithReasoning(message, usage, reasoning, reasoningDetails, toolCalls...), nil
}

// ChatCompletionStream implements llm.ChatCompletionStreamingClient.
func (c *ChatCompletionClient) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	// Validate options before proceeding
	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	temperature := float32(opts.Temperature)

	req := openrouter.ChatCompletionRequest{
		Model:       c.model,
		Temperature: temperature,
		Stream:      true, // Enable streaming
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
				Strict:      true,
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

	req.ToolChoice = opts.ToolChoice

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

	// Create streaming channel
	chunks := make(chan llm.StreamChunk, 10)

	var (
		promptTokens     atomic.Int64
		completionTokens atomic.Int64
		totalTokens      atomic.Int64
	)

	go func() {
		defer close(chunks)

		stream, err := c.client.CreateChatCompletionStream(ctx, req)
		if err != nil {
			var reqErr *openrouter.RequestError
			if errors.As(err, &reqErr) {
				chunks <- llm.NewErrorStreamChunk(errors.WithStack(llm.NewHTTPError(reqErr.HTTPStatusCode, reqErr.Error())))
				return
			}
			chunks <- llm.NewErrorStreamChunk(errors.WithStack(err))
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
				chunks <- llm.NewErrorStreamChunk(errors.WithStack(err))
				return
			}

			if len(response.Choices) == 0 {
				continue
			}

			if response.Usage != nil {
				promptTokens.Store(int64(response.Usage.PromptTokens))
				completionTokens.Store(int64(response.Usage.CompletionTokens))
				totalTokens.Store(int64(response.Usage.TotalTokens))
			}

			choice := response.Choices[0]
			delta := choice.Delta

			// Create stream delta
			var toolCallDeltas []llm.ToolCallDelta
			for i, tc := range delta.ToolCalls {
				toolCallDeltas = append(toolCallDeltas, llm.NewToolCallDelta(
					i,
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

			if deltaReasoning != "" || len(deltaReasoningDetails) > 0 {
				streamDelta := llm.NewReasoningStreamDelta(
					llm.RoleAssistant,
					delta.Content,
					deltaReasoning,
					deltaReasoningDetails,
					toolCallDeltas...,
				)
				chunks <- llm.NewStreamChunk(streamDelta)
			} else {
				streamDelta := llm.NewStreamDelta(
					llm.RoleAssistant,
					delta.Content,
					toolCallDeltas...,
				)
				chunks <- llm.NewStreamChunk(streamDelta)
			}
		}

		// Send completion chunk with usage if available
		usage := llm.NewChatCompletionUsage(
			promptTokens.Load(),
			completionTokens.Load(),
			totalTokens.Load(),
		)

		chunks <- llm.NewCompleteStreamChunk(usage)
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

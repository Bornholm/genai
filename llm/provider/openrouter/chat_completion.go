package openrouter

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/context"
	"github.com/pkg/errors"
	"github.com/revrost/go-openrouter"
)

type ChatCompletionClient struct {
	client *openrouter.Client
	model  string
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

	messages := make([]openrouter.ChatCompletionMessage, 0, len(opts.Messages))

	// Create validator for provider-specific validation (Layer 2)
	validator := NewOpenRouterAttachmentValidator(c.model)

	for _, m := range opts.Messages {
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
			messages = append(messages, openrouter.ChatCompletionMessage{
				Role: openrouter.ChatMessageRoleAssistant,
				Content: openrouter.Content{
					Text: m.Content(),
				},
			})
		case llm.RoleTool:
			if len(m.Attachments()) > 0 {
				return nil, errors.Errorf("tool messages cannot have attachments")
			}
			toolMessage, ok := m.(llm.ToolMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool message type '%T'", m)
			}

			messages = append(messages, openrouter.ChatCompletionMessage{
				Role:       openrouter.ChatMessageRoleTool,
				ToolCallID: toolMessage.ID(),
				Content: openrouter.Content{
					Text: m.Content(),
				},
			})
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

			messages = append(messages, message)
		}
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
			if reqErr.HTTPStatusCode == http.StatusTooManyRequests {
				return nil, errors.WithStack(llm.ErrRateLimit)
			}
		}

		return nil, errors.WithStack(err)
	}

	if len(res.Choices) == 0 {
		return nil, errors.WithStack(llm.ErrNoMessage)
	}

	openrouterMessage := res.Choices[0].Message

	var message llm.Message = llm.NewMessage(llm.RoleAssistant, openrouterMessage.Content.Text)

	toolCalls := make([]llm.ToolCall, 0)

	for _, tc := range openrouterMessage.ToolCalls {
		toolCalls = append(toolCalls, llm.NewToolCall(tc.ID, tc.Function.Name, tc.Function.Arguments))
	}

	usage := llm.NewChatCompletionUsage(int64(res.Usage.PromptTokens), int64(res.Usage.CompletionTokens), int64(res.Usage.TotalTokens))

	return llm.NewChatCompletionResponse(message, usage, toolCalls...), nil
}

func NewChatCompletionClient(client *openrouter.Client, model string) *ChatCompletionClient {
	return &ChatCompletionClient{
		client: client,
		model:  model,
	}
}

var _ llm.ChatCompletionClient = &ChatCompletionClient{}

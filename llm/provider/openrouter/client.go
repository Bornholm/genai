package openrouter

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"github.com/revrost/go-openrouter"
)

type Client struct {
	client *openrouter.Client
	model  string
}

// Model implements llm.Client.
func (c *Client) Model() string {
	return c.model
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.CompletionResponse, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	req := openrouter.ChatCompletionRequest{
		Model:       c.model,
		Temperature: float32(opts.Temperature),
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

	for _, m := range opts.Messages {
		switch m.Role() {
		case llm.RoleSystem:
			messages = append(messages, openrouter.ChatCompletionMessage{
				Role:    openrouter.ChatMessageRoleSystem,
				Content: m.Content(),
			})
		case llm.RoleUser:
			messages = append(messages, openrouter.ChatCompletionMessage{
				Role:    openrouter.ChatMessageRoleUser,
				Content: m.Content(),
			})
		case llm.RoleAssistant:
			messages = append(messages, openrouter.ChatCompletionMessage{
				Role:    openrouter.ChatMessageRoleAssistant,
				Content: m.Content(),
			})
		case llm.RoleTool:
			toolMessage, ok := m.(llm.ToolMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool message type '%T'", m)
			}

			messages = append(messages, openrouter.ChatCompletionMessage{
				Role:       openrouter.ChatMessageRoleTool,
				ToolCallID: toolMessage.ID(),
				Content:    m.Content(),
			})
		case llm.RoleToolCalls:
			toolCallsMessage, ok := m.(llm.ToolCallsMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool calls message type '%T'", m)
			}

			message := openrouter.ChatCompletionMessage{
				Role: openrouter.ChatMessageRoleAssistant,
			}

			toolCalls := make([]openrouter.ToolCall, 0, len(toolCallsMessage.ToolCalls()))
			for _, tc := range toolCallsMessage.ToolCalls() {
				toolCalls = append(toolCalls, openrouter.ToolCall{
					ID: tc.ID(),
					Function: openrouter.FunctionCall{
						Name:      tc.Name(),
						Arguments: tc.Parameters().(string),
					},
					Type: openrouter.ToolTypeFunction,
				})
			}
			message.ToolCalls = toolCalls

			messages = append(messages, message)
		}
	}

	req.Messages = messages

	res, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if len(res.Choices) == 0 {
		return nil, errors.WithStack(llm.ErrNoMessage)
	}

	openrouterMessage := res.Choices[0].Message

	var message llm.Message = llm.NewMessage(llm.RoleAssistant, openrouterMessage.Content)

	toolCalls := make([]llm.ToolCall, 0)

	for _, tc := range openrouterMessage.ToolCalls {
		toolCalls = append(toolCalls, llm.NewToolCall(tc.ID, tc.Function.Name, tc.Function.Arguments))
	}

	return llm.NewCompletionResponse(message, toolCalls...), nil
}

func NewClient(client *openrouter.Client, model string) *Client {
	return &Client{
		client: client,
		model:  model,
	}
}

var _ llm.Client = &Client{}

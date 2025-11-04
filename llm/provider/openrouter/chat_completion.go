package openrouter

import (
	"net/http"

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

	temperature := float32(opts.Temperature)

	req := openrouter.ChatCompletionRequest{
		Model:       c.model,
		Temperature: &temperature,
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
				Role: openrouter.ChatMessageRoleSystem,
				Content: openrouter.Content{
					Text: m.Content(),
				},
			})
		case llm.RoleUser:
			messages = append(messages, openrouter.ChatCompletionMessage{
				Role: openrouter.ChatMessageRoleUser,
				Content: openrouter.Content{
					Text: m.Content(),
				},
			})
		case llm.RoleAssistant:
			messages = append(messages, openrouter.ChatCompletionMessage{
				Role: openrouter.ChatMessageRoleAssistant,
				Content: openrouter.Content{
					Text: m.Content(),
				},
			})
		case llm.RoleTool:
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

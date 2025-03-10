package openai

import (
	"context"
	"log"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"github.com/pkg/errors"
)

type Client struct {
	client              *openai.Client
	chatCompletionModel string
	embeddingsModel     string
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.CompletionResponse, error) {
	if c.chatCompletionModel == "" {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	opts := llm.NewChatCompletionOptions(funcs...)

	params := openai.ChatCompletionNewParams{
		Model:       openai.F(openai.ChatModel(c.chatCompletionModel)),
		Temperature: openai.Float(opts.Temperature),
	}

	if opts.ResponseFormat == llm.ResponseFormatJSON {
		jsonFormat := openai.ResponseFormatJSONSchemaParam{
			Type: openai.F(openai.ResponseFormatJSONSchemaTypeJSONSchema),
		}

		if opts.ResponseSchema != nil {
			jsonFormat.JSONSchema = openai.F(openai.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        openai.F(opts.ResponseSchema.Name()),
				Description: openai.F(opts.ResponseSchema.Description()),
				Schema:      openai.F(opts.ResponseSchema.Schema()),
				Strict:      openai.Bool(true),
			})
		}

		params.ResponseFormat = openai.F[openai.ChatCompletionNewParamsResponseFormatUnion](jsonFormat)
	}

	if len(opts.Tools) > 0 {
		tools := make([]openai.ChatCompletionToolParam, 0, len(opts.Tools))

		for _, t := range opts.Tools {
			tools = append(tools, openai.ChatCompletionToolParam{
				Type: openai.F(openai.ChatCompletionToolTypeFunction),
				Function: openai.F(openai.FunctionDefinitionParam{
					Name:        openai.String(t.Name()),
					Description: openai.String(t.Description()),
					Parameters:  openai.F(shared.FunctionParameters(t.Parameters())),
				}),
			})
		}

		params.Tools = openai.F(tools)
	}

	if opts.ToolChoice != llm.ToolChoiceDefault {
		switch opts.ToolChoice {
		case llm.ToolChoiceAuto:
			params.ToolChoice = openai.F[openai.ChatCompletionToolChoiceOptionUnionParam](
				openai.ChatCompletionToolChoiceOptionAutoAuto,
			)
		}
	}

	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(opts.Messages))

	for _, m := range opts.Messages {
		switch m.Role() {
		case llm.RoleSystem:
			messages = append(messages, openai.SystemMessage(m.Content()))
		case llm.RoleUser:
			messages = append(messages, openai.UserMessage(m.Content()))
		case llm.RoleAssistant:
			messages = append(messages, openai.AssistantMessage(m.Content()))
		case llm.RoleTool:
			toolMessage, ok := m.(llm.ToolMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool message type '%T'", m)
			}

			messages = append(messages, openai.ToolMessage(toolMessage.ID(), toolMessage.Content()))
		case llm.RoleToolCalls:
			toolCallsMessage, ok := m.(llm.ToolCallsMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool calls message type '%T'", m)
			}

			message := openai.ChatCompletionAssistantMessageParam{
				Role: openai.F(openai.ChatCompletionAssistantMessageParamRoleAssistant),
			}

			toolCalls := make([]openai.ChatCompletionMessageToolCallParam, 0, len(toolCallsMessage.ToolCalls()))
			for _, tc := range toolCallsMessage.ToolCalls() {
				toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
					ID: openai.F(tc.ID()),
					Function: openai.F(openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      openai.F(tc.Name()),
						Arguments: openai.F(tc.Parameters().(string)),
					}),
					Type: openai.F(openai.ChatCompletionMessageToolCallTypeFunction),
				})
			}
			message.ToolCalls = openai.F(toolCalls)

			messages = append(messages, message)
		}
	}

	params.Messages = openai.F(messages)

	log.Printf("New completion request with %d messages", len(messages))

	completion, err := c.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if len(completion.Choices) == 0 {
		return nil, errors.WithStack(llm.ErrNoMessage)
	}

	openaiMessage := completion.Choices[0].Message

	var message llm.Message = llm.NewMessage(llm.RoleAssistant, openaiMessage.Content)

	toolCalls := make([]llm.ToolCall, 0)

	for _, tc := range openaiMessage.ToolCalls {
		toolCalls = append(toolCalls, llm.NewToolCall(tc.ID, tc.Function.Name, tc.Function.Arguments))
	}

	return llm.NewCompletionResponse(message, toolCalls...), nil
}

func NewClient(client *openai.Client, chatCompletionModel string, embeddingsModel string) *Client {
	return &Client{
		client:              client,
		chatCompletionModel: chatCompletionModel,
		embeddingsModel:     embeddingsModel,
	}
}

var _ llm.Client = &Client{}

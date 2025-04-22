package openai

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"github.com/pkg/errors"
)

type ChatCompletionClient struct {
	client openai.Client
	model  string
}

// ChatCompletion implements llm.Client.
func (c *ChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	if c.model == "" {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	opts := llm.NewChatCompletionOptions(funcs...)

	params := openai.ChatCompletionNewParams{
		Model:       openai.ChatModel(c.model),
		Temperature: openai.Float(opts.Temperature),
	}

	if opts.ResponseFormat == llm.ResponseFormatJSON {
		jsonFormat := openai.ResponseFormatJSONSchemaParam{}

		if opts.ResponseSchema != nil {
			jsonFormat.JSONSchema = openai.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        opts.ResponseSchema.Name(),
				Description: openai.Opt(opts.ResponseSchema.Description()),
				Schema:      opts.ResponseSchema.Schema(),
				Strict:      openai.Bool(true),
			}
		}

		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: jsonFormat.JSONSchema,
			},
		}
	}

	if len(opts.Tools) > 0 {
		tools := make([]openai.ChatCompletionToolParam, 0, len(opts.Tools))

		for _, t := range opts.Tools {
			tools = append(tools, openai.ChatCompletionToolParam{
				Function: openai.FunctionDefinitionParam{
					Name:        t.Name(),
					Description: openai.String(t.Description()),
					Parameters:  shared.FunctionParameters(t.Parameters()),
				},
			})
		}

		params.Tools = tools
	}

	if opts.ToolChoice != llm.ToolChoiceDefault {
		switch opts.ToolChoice {
		case llm.ToolChoiceAuto:
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.Opt(string(openai.ChatCompletionToolChoiceOptionAutoAuto)),
			}
		}
	}

	if opts.Seed != nil {
		params.Seed = openai.Int(int64(*opts.Seed))
	}

	if opts.MaxCompletionTokens != nil {
		params.MaxCompletionTokens = openai.Int(int64(*opts.MaxCompletionTokens))
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

			message := openai.ChatCompletionAssistantMessageParam{}

			toolCalls := make([]openai.ChatCompletionMessageToolCallParam, 0, len(toolCallsMessage.ToolCalls()))
			for _, tc := range toolCallsMessage.ToolCalls() {
				toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
					ID: tc.ID(),
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      tc.Name(),
						Arguments: tc.Parameters().(string),
					},
				})
			}
			message.ToolCalls = toolCalls

			messages = append(messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &message})
		}
	}

	params.Messages = messages

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

	usage := llm.NewChatCompletionUsage(completion.Usage.PromptTokens, completion.Usage.CompletionTokens, completion.Usage.TotalTokens)

	return llm.NewChatCompletionResponse(message, usage, toolCalls...), nil
}

func NewChatCompletionClient(client openai.Client, model string) *ChatCompletionClient {
	return &ChatCompletionClient{
		client: client,
		model:  model,
	}
}

var _ llm.ChatCompletionClient = &ChatCompletionClient{}

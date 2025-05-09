package openai

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"github.com/pkg/errors"
)

type ParamsBuilder interface {
	BuildParams(ctx context.Context, opts *llm.ChatCompletionOptions) (*openai.ChatCompletionNewParams, error)
}

type BuildParamsFunc func(ctx context.Context, opts *llm.ChatCompletionOptions) (*openai.ChatCompletionNewParams, error)

func (fn BuildParamsFunc) BuildParams(ctx context.Context, opts *llm.ChatCompletionOptions) (*openai.ChatCompletionNewParams, error) {
	return fn(ctx, opts)
}

type paramsBuilder struct {
	model string
}

func (b *paramsBuilder) BuildParams(ctx context.Context, opts *llm.ChatCompletionOptions) (*openai.ChatCompletionNewParams, error) {
	if b.model == "" {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	params, err := ConfigureParams(
		ctx, opts,
		ConfigureTools,
		ConfigureTemperature,
		ConfigureResponseFormat,
		ConfigureMessages,
		ConfigureMaxCompletionTokens,
		ConfigureSeed,
	)
	if err != nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	params.Model = openai.ChatModel(b.model)

	return params, nil
}

var _ ParamsBuilder = &paramsBuilder{}

type ConfigureParamsFunc func(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error

func ConfigureParams(ctx context.Context, opts *llm.ChatCompletionOptions, funcs ...ConfigureParamsFunc) (*openai.ChatCompletionNewParams, error) {
	params := &openai.ChatCompletionNewParams{}
	for _, fn := range funcs {
		if err := fn(ctx, opts, params); err != nil {
			return nil, errors.WithStack(err)
		}
	}
	return params, nil
}

func ConfigureTemperature(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
	params.Temperature = openai.Float(opts.Temperature)
	return nil
}

func ConfigureMaxCompletionTokens(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
	if opts.MaxCompletionTokens == nil {
		return nil
	}

	params.MaxCompletionTokens = openai.Int(int64(*opts.MaxCompletionTokens))

	return nil
}

func ConfigureResponseFormat(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
	if opts.ResponseFormat != llm.ResponseFormatJSON {
		return nil
	}

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

	return nil
}

func ConfigureSeed(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
	if opts.Seed == nil {
		return nil
	}

	params.Seed = openai.Int(int64(*opts.Seed))
	return nil
}

func ConfigureTools(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
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

	return nil
}

func ConfigureMessages(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
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
				return errors.Errorf("unexpected tool message type '%T'", m)
			}

			messages = append(messages, openai.ToolMessage(toolMessage.Content(), toolMessage.ID()))
		case llm.RoleToolCalls:
			toolCallsMessage, ok := m.(llm.ToolCallsMessage)
			if !ok {
				return errors.Errorf("unexpected tool calls message type '%T'", m)
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

	return nil
}

package codec

import (
	"context"

	"github.com/bornholm/genai/llm"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/pkg/errors"
)

// ToolToProto converts a tool descriptor. Only the description travels: the
// plugin never executes a tool, the host does.
func ToolToProto(tool llm.Tool) (*pluginv1.Tool, error) {
	params, err := structFromAny(tool.Parameters())
	if err != nil {
		return nil, errors.Wrapf(err, "could not encode parameters of tool %q", tool.Name())
	}
	return &pluginv1.Tool{
		Name:        tool.Name(),
		Description: tool.Description(),
		Parameters:  params,
	}, nil
}

// ErrToolNotExecutable is returned when a provider running inside a plugin
// tries to execute a tool it only holds a descriptor of.
var ErrToolNotExecutable = errors.New("tool descriptors received over the plugin protocol cannot be executed")

// ToolFromProto rebuilds a descriptor-only tool.
func ToolFromProto(tool *pluginv1.Tool) llm.Tool {
	var params map[string]any
	if tool.GetParameters() != nil {
		params = tool.GetParameters().AsMap()
	}
	return llm.NewFuncTool(tool.GetName(), tool.GetDescription(), params, func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
		return nil, errors.WithStack(ErrToolNotExecutable)
	})
}

// ResponseSchemaToProto converts a response schema.
func ResponseSchemaToProto(schema llm.ResponseSchema) (*pluginv1.ResponseSchema, error) {
	if schema == nil {
		return nil, nil
	}
	value, err := valueFromAny(schema.Schema())
	if err != nil {
		return nil, errors.Wrapf(err, "could not encode response schema %q", schema.Name())
	}
	return &pluginv1.ResponseSchema{
		Name:        schema.Name(),
		Description: schema.Description(),
		Schema:      value,
		Strict:      llm.IsStrictResponseSchema(schema),
	}, nil
}

// ResponseSchemaFromProto converts a response schema.
func ResponseSchemaFromProto(schema *pluginv1.ResponseSchema) llm.ResponseSchema {
	if schema == nil {
		return nil
	}
	var value any
	if schema.GetSchema() != nil {
		value = schema.GetSchema().AsInterface()
	}
	return llm.NewResponseSchema(schema.GetName(), schema.GetDescription(), value).WithStrict(schema.GetStrict())
}

// ReasoningOptionsToProto converts reasoning options.
func ReasoningOptionsToProto(opts *llm.ReasoningOptions) *pluginv1.ReasoningOptions {
	if opts == nil {
		return nil
	}
	result := &pluginv1.ReasoningOptions{Exclude: opts.Exclude}
	if opts.Effort != nil {
		effort := string(*opts.Effort)
		result.Effort = &effort
	}
	if opts.MaxTokens != nil {
		maxTokens := int64(*opts.MaxTokens)
		result.MaxTokens = &maxTokens
	}
	if opts.Enabled != nil {
		enabled := *opts.Enabled
		result.Enabled = &enabled
	}
	return result
}

// ReasoningOptionsFromProto converts reasoning options.
func ReasoningOptionsFromProto(opts *pluginv1.ReasoningOptions) *llm.ReasoningOptions {
	if opts == nil {
		return nil
	}
	result := &llm.ReasoningOptions{Exclude: opts.GetExclude()}
	if opts.Effort != nil {
		effort := llm.ReasoningEffort(opts.GetEffort())
		result.Effort = &effort
	}
	if opts.MaxTokens != nil {
		maxTokens := int(opts.GetMaxTokens())
		result.MaxTokens = &maxTokens
	}
	if opts.Enabled != nil {
		enabled := opts.GetEnabled()
		result.Enabled = &enabled
	}
	return result
}

// ChatCompletionOptionsToProto converts resolved chat completion options.
func ChatCompletionOptionsToProto(opts *llm.ChatCompletionOptions) (*pluginv1.ChatCompletionOptions, error) {
	messages, err := messagesToProto(opts.Messages)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	var tools []*pluginv1.Tool
	if len(opts.Tools) > 0 {
		tools = make([]*pluginv1.Tool, 0, len(opts.Tools))
		for _, tool := range opts.Tools {
			converted, err := ToolToProto(tool)
			if err != nil {
				return nil, errors.WithStack(err)
			}
			tools = append(tools, converted)
		}
	}

	schema, err := ResponseSchemaToProto(opts.ResponseSchema)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	extra, err := structFromAny(opts.ExtraFields)
	if err != nil {
		return nil, errors.Wrap(err, "could not encode extra fields")
	}

	result := &pluginv1.ChatCompletionOptions{
		Messages:       messages,
		Tools:          tools,
		ToolChoice:     string(opts.ToolChoice),
		Temperature:    opts.Temperature,
		ResponseFormat: string(opts.ResponseFormat),
		ResponseSchema: schema,
		Reasoning:      ReasoningOptionsToProto(opts.Reasoning),
		Modalities:     opts.Modalities,
		SessionId:      opts.SessionID,
		ExtraFields:    extra,
	}
	if opts.Seed != nil {
		seed := int64(*opts.Seed)
		result.Seed = &seed
	}
	if opts.MaxCompletionTokens != nil {
		maxTokens := int64(*opts.MaxCompletionTokens)
		result.MaxCompletionTokens = &maxTokens
	}
	if opts.Audio != nil {
		result.Audio = &pluginv1.AudioOutputConfig{Voice: opts.Audio.Voice, Format: opts.Audio.Format}
	}
	return result, nil
}

// ChatCompletionOptionsFromProto rebuilds the option funcs that produce the
// same llm.ChatCompletionOptions on the other side.
//
// ExtraFields travel as a protobuf Struct, so every number comes back as a
// float64 whatever its Go type was on the host: a plugin reads them the way
// it would read decoded JSON, not with an int type assertion.
func ChatCompletionOptionsFromProto(opts *pluginv1.ChatCompletionOptions) ([]llm.ChatCompletionOptionFunc, error) {
	messages, err := messagesFromProto(opts.GetMessages())
	if err != nil {
		return nil, errors.WithStack(err)
	}

	funcs := []llm.ChatCompletionOptionFunc{
		llm.WithMessages(messages...),
	}

	if len(opts.GetTools()) > 0 {
		tools := make([]llm.Tool, 0, len(opts.GetTools()))
		for _, tool := range opts.GetTools() {
			tools = append(tools, ToolFromProto(tool))
		}
		funcs = append(funcs, llm.WithTools(tools...))
	}
	if opts.GetToolChoice() != "" {
		funcs = append(funcs, llm.WithToolChoice(llm.ToolChoice(opts.GetToolChoice())))
	}
	if opts.Temperature != nil {
		funcs = append(funcs, llm.WithTemperature(opts.GetTemperature()))
	}
	if opts.GetResponseFormat() != "" {
		funcs = append(funcs, llm.WithResponseFormat(llm.ResponseFormat(opts.GetResponseFormat())))
	}
	if schema := ResponseSchemaFromProto(opts.GetResponseSchema()); schema != nil {
		funcs = append(funcs, llm.WithResponseSchema(schema))
	}
	if opts.Seed != nil {
		funcs = append(funcs, llm.WithSeed(int(opts.GetSeed())))
	}
	if opts.MaxCompletionTokens != nil {
		funcs = append(funcs, llm.WithMaxCompletionTokens(int(opts.GetMaxCompletionTokens())))
	}
	if reasoning := ReasoningOptionsFromProto(opts.GetReasoning()); reasoning != nil {
		funcs = append(funcs, llm.WithReasoning(reasoning))
	}
	if len(opts.GetModalities()) > 0 {
		funcs = append(funcs, llm.WithModalities(opts.GetModalities()...))
	}
	if opts.GetAudio() != nil {
		funcs = append(funcs, llm.WithAudioOutput(opts.GetAudio().GetVoice(), opts.GetAudio().GetFormat()))
	}
	if opts.GetSessionId() != "" {
		funcs = append(funcs, llm.WithSessionID(opts.GetSessionId()))
	}
	if opts.GetExtraFields() != nil {
		funcs = append(funcs, llm.WithExtraFields(opts.GetExtraFields().AsMap()))
	}

	return funcs, nil
}

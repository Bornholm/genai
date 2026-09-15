package anthropic

import (
	"encoding/json"
	"fmt"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// minThinkingBudget is the smallest budget_tokens the Messages API accepts.
const minThinkingBudget int64 = 1024

// effortRatios maps an OpenAI-style reasoning effort to the share of
// max_tokens granted to thinking, as documented on llm.ReasoningEffort.
var effortRatios = map[llm.ReasoningEffort]float64{
	llm.ReasoningEffortXHigh:   0.95,
	llm.ReasoningEffortHigh:    0.80,
	llm.ReasoningEffortMedium:  0.50,
	llm.ReasoningEffortLow:     0.20,
	llm.ReasoningEffortMinimal: 0.10,
}

// buildParams converts genai chat completion options to Messages API
// parameters.
func buildParams(opts *llm.ChatCompletionOptions, model string, defaultMaxTokens int64) (*anthropicsdk.MessageNewParams, error) {
	params := &anthropicsdk.MessageNewParams{
		Model:     model,
		MaxTokens: defaultMaxTokens,
	}

	if opts.MaxCompletionTokens != nil {
		params.MaxTokens = int64(*opts.MaxCompletionTokens)
	}

	thinking, err := configureThinking(params, opts.Reasoning, opts.MaxCompletionTokens != nil)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	// The API rejects any temperature but the default once thinking is on;
	// dropping it beats a 400 the caller cannot act on.
	if opts.Temperature != nil && !thinking {
		params.Temperature = anthropicsdk.Float(*opts.Temperature)
	}

	if len(opts.Tools) > 0 {
		tools := make([]anthropicsdk.ToolUnionParam, 0, len(opts.Tools))
		for _, t := range opts.Tools {
			tools = append(tools, anthropicsdk.ToolUnionParam{OfTool: toolParam(t)})
		}
		params.Tools = tools

		switch opts.ToolChoice {
		case llm.ToolChoiceAuto:
			params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfAuto: &anthropicsdk.ToolChoiceAutoParam{}}
		case llm.ToolChoiceRequired:
			params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfAny: &anthropicsdk.ToolChoiceAnyParam{}}
		case llm.ToolChoiceNone:
			params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfNone: &anthropicsdk.ToolChoiceNoneParam{}}
		}
	}

	// Structured output is only expressible with a schema: the Messages API
	// has no schema-less "json mode". Failing loudly beats handing free text
	// to a caller that will try to parse it.
	if opts.ResponseFormat == llm.ResponseFormatJSON {
		if opts.ResponseSchema == nil {
			return nil, llm.NewValidationError("response_schema", "the anthropic provider requires a response schema for the JSON response format")
		}
		schema, err := toMap(opts.ResponseSchema.Schema())
		if err != nil {
			return nil, errors.Wrap(err, "could not convert response schema")
		}
		params.OutputConfig.Format = anthropicsdk.JSONOutputFormatParam{Schema: schema}
	}

	system, messages, err := buildMessages(opts.Messages)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	params.System = system
	params.Messages = messages

	if len(opts.ExtraFields) > 0 {
		params.SetExtraFields(opts.ExtraFields)
	}

	return params, nil
}

// configureThinking maps llm.ReasoningOptions to the thinking configuration
// and reports whether thinking ended up enabled.
//
// An explicit MaxTokens becomes budget_tokens verbatim; an effort level is
// converted to a share of max_tokens. The API requires
// minThinkingBudget <= budget_tokens < max_tokens. When the caller chose
// max_tokens, that ceiling is respected and the budget is clamped under it,
// with an error when the minimum cannot fit; when max_tokens is only the
// provider default, it is raised instead so the budget the caller asked for
// is honoured.
func configureThinking(params *anthropicsdk.MessageNewParams, reasoning *llm.ReasoningOptions, explicitMaxTokens bool) (bool, error) {
	if reasoning == nil {
		return false, nil
	}
	if reasoning.Enabled != nil && !*reasoning.Enabled {
		return false, nil
	}
	if reasoning.Effort != nil && *reasoning.Effort == llm.ReasoningEffortNone {
		return false, nil
	}

	var budget int64
	switch {
	case reasoning.MaxTokens != nil:
		budget = int64(*reasoning.MaxTokens)
	case reasoning.Effort != nil:
		ratio, known := effortRatios[*reasoning.Effort]
		if !known {
			ratio = effortRatios[llm.ReasoningEffortMedium]
		}
		budget = int64(float64(params.MaxTokens) * ratio)
	case reasoning.Enabled != nil && *reasoning.Enabled:
		budget = int64(float64(params.MaxTokens) * effortRatios[llm.ReasoningEffortMedium])
	default:
		return false, nil
	}

	if budget < minThinkingBudget {
		budget = minThinkingBudget
	}
	if params.MaxTokens <= budget {
		if explicitMaxTokens {
			budget = params.MaxTokens - 1
			if budget < minThinkingBudget {
				return false, llm.NewValidationError("max_completion_tokens",
					fmt.Sprintf("max completion tokens must exceed %d to leave room for reasoning", minThinkingBudget))
			}
		} else {
			params.MaxTokens = budget + DefaultMaxTokens
		}
	}

	params.Thinking = anthropicsdk.ThinkingConfigParamOfEnabled(budget)

	return true, nil
}

// toolParam converts an llm.Tool to a Messages API tool definition. The
// JSON schema returned by Parameters() is split into the typed
// properties/required fields, every other keyword travelling as an extra
// field so nothing is lost.
func toolParam(t llm.Tool) *anthropicsdk.ToolParam {
	tool := &anthropicsdk.ToolParam{
		Name: t.Name(),
	}
	if description := t.Description(); description != "" {
		tool.Description = anthropicsdk.String(description)
	}

	schema := anthropicsdk.ToolInputSchemaParam{}
	extra := map[string]any{}
	for key, value := range t.Parameters() {
		switch key {
		case "type":
			// Always "object" for a tool input schema.
		case "properties":
			schema.Properties = value
		case "required":
			schema.Required = toStringSlice(value)
		default:
			extra[key] = value
		}
	}
	if schema.Properties == nil {
		schema.Properties = map[string]any{}
	}
	if len(extra) > 0 {
		schema.ExtraFields = extra
	}
	tool.InputSchema = schema

	return tool
}

func toStringSlice(value any) []string {
	switch typed := value.(type) {
	case []string:
		return typed
	case []any:
		result := make([]string, 0, len(typed))
		for _, item := range typed {
			if s, ok := item.(string); ok {
				result = append(result, s)
			}
		}
		return result
	}
	return nil
}

// toMap round-trips an arbitrary schema value through JSON to obtain the
// map[string]any the SDK expects.
func toMap(value any) (map[string]any, error) {
	if m, ok := value.(map[string]any); ok {
		return m, nil
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, errors.WithStack(err)
	}
	return m, nil
}

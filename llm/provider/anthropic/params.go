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

// minOutputMargin is the smallest number of tokens kept for the visible
// answer when a thinking budget has to fit under a caller-chosen max_tokens.
const minOutputMargin int64 = 1024

// maxTemperature is the upper bound of the Messages API temperature range.
const maxTemperature = 1.0

// jsonModeInstruction stands in for a schema-less JSON response format.
const jsonModeInstruction = "Respond with a single valid JSON object and nothing else: no prose, no markdown fences."

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

	thinking, err := configureThinking(params, opts.Reasoning, opts.MaxCompletionTokens != nil, defaultMaxTokens)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	// The API rejects any temperature but the default once thinking is on;
	// dropping it beats a 400 the caller cannot act on. It also caps the
	// range at 1 where the shared options allow up to 2 (the OpenAI range),
	// so a value above it is clamped rather than refused upstream.
	if opts.Temperature != nil && !thinking {
		params.Temperature = anthropicsdk.Float(min(*opts.Temperature, maxTemperature))
	}

	if len(opts.Tools) > 0 {
		tools := make([]anthropicsdk.ToolUnionParam, 0, len(opts.Tools))
		for _, t := range opts.Tools {
			tool, err := toolParam(t)
			if err != nil {
				return nil, errors.Wrapf(err, "invalid tool '%s'", t.Name())
			}
			tools = append(tools, anthropicsdk.ToolUnionParam{OfTool: tool})
		}
		params.Tools = tools

		switch opts.ToolChoice {
		case llm.ToolChoiceAuto:
			params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfAuto: &anthropicsdk.ToolChoiceAutoParam{}}
		case llm.ToolChoiceRequired:
			// With extended thinking the API only accepts auto or none;
			// falling back to auto beats a 400 the caller cannot act on.
			if thinking {
				params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfAuto: &anthropicsdk.ToolChoiceAutoParam{}}
			} else {
				params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfAny: &anthropicsdk.ToolChoiceAnyParam{}}
			}
		case llm.ToolChoiceNone:
			params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfNone: &anthropicsdk.ToolChoiceNoneParam{}}
		}
	}

	// Structured output is only expressible with a schema: the Messages API
	// has no schema-less "json mode". Without one (OpenAI's json_object,
	// which the proxy forwards as a bare JSON format) the model is instructed
	// through the system prompt instead, the way the OpenAI mode itself
	// expects the prompt to ask for JSON.
	jsonMode := opts.ResponseFormat == llm.ResponseFormatJSON
	if jsonMode && opts.ResponseSchema != nil {
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
	if len(messages) == 0 {
		return nil, llm.NewValidationError("messages", "at least one non-system message is required")
	}
	if jsonMode && opts.ResponseSchema == nil {
		system = append(system, anthropicsdk.TextBlockParam{Text: jsonModeInstruction})
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
// minThinkingBudget <= budget_tokens < max_tokens, and an answer needs room
// past the budget: an output margin of a quarter of max_tokens, at least
// minOutputMargin, is always kept. A budget derived from an effort is a
// share of max_tokens by definition, so it is clamped under the margin
// whatever max_tokens is; an explicit budget is honoured by raising a
// max_tokens that is only the provider default, and clamped under a
// caller-chosen one, with an error when margin and minimum budget cannot
// both fit. A caller asking for a large explicit budget should set
// max_tokens itself: the raised value is not checked against the model's
// output limit.
func configureThinking(params *anthropicsdk.MessageNewParams, reasoning *llm.ReasoningOptions, explicitMaxTokens bool, defaultMaxTokens int64) (bool, error) {
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

	margin := max(params.MaxTokens/4, minOutputMargin)
	explicitBudget := reasoning.MaxTokens != nil
	switch {
	case explicitMaxTokens || !explicitBudget:
		if budget > params.MaxTokens-margin {
			budget = params.MaxTokens - margin
		}
		if budget < minThinkingBudget {
			field, source := "max_completion_tokens", "max completion tokens"
			if !explicitMaxTokens {
				field, source = "max_tokens", "the provider's MAX_TOKENS default"
			}
			return false, llm.NewValidationError(field,
				fmt.Sprintf("%s must be at least %d to hold a reasoning budget and an answer", source, minThinkingBudget+minOutputMargin))
		}
	case budget > params.MaxTokens-margin:
		// The default is ours to grow, up to what the answer needs: the
		// smallest raise keeps the result under the output limit of every
		// model that accepts the budget itself.
		params.MaxTokens = budget + min(defaultMaxTokens, max(budget/4, minOutputMargin))
	}

	params.Thinking = anthropicsdk.ThinkingConfigParamOfEnabled(budget)

	return true, nil
}

// toolParam converts an llm.Tool to a Messages API tool definition. The
// JSON schema returned by Parameters() is split into the typed
// properties/required fields, every other keyword travelling as an extra
// field so nothing is lost.
func toolParam(t llm.Tool) (*anthropicsdk.ToolParam, error) {
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
			// The API only takes object schemas; anything else would be
			// silently rewritten as one by the SDK's default.
			if typ, _ := value.(string); typ != "" && typ != "object" {
				return nil, errors.Errorf("tool input schema must be of type object, got %q", typ)
			}
		case "properties":
			schema.Properties = value
		case "required":
			required, err := toStringSlice(value)
			if err != nil {
				return nil, errors.Wrap(err, "invalid 'required' keyword")
			}
			schema.Required = required
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

	return tool, nil
}

// toStringSlice decodes a JSON schema "required" array. A malformed entry
// is an error rather than a silently dropped requirement: the model would
// otherwise be told a mandatory argument is optional.
func toStringSlice(value any) ([]string, error) {
	switch typed := value.(type) {
	case nil:
		return nil, nil
	case []string:
		return typed, nil
	case []any:
		result := make([]string, 0, len(typed))
		for _, item := range typed {
			s, ok := item.(string)
			if !ok {
				return nil, errors.Errorf("expected a string, got %T", item)
			}
			result = append(result, s)
		}
		return result, nil
	}
	return nil, errors.Errorf("expected an array of strings, got %T", value)
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

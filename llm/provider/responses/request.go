package responses

import (
	"encoding/json"
	"fmt"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// request is the body of POST /responses. Only the fields genai has an option
// for are typed; ExtraFields travel through MarshalJSON.
type request struct {
	Model           string         `json:"model"`
	Input           []inputItem    `json:"input"`
	Tools           []tool         `json:"tools,omitempty"`
	ToolChoice      string         `json:"tool_choice,omitempty"`
	Temperature     *float64       `json:"temperature,omitempty"`
	MaxOutputTokens *int           `json:"max_output_tokens,omitempty"`
	Reasoning       *reasoning     `json:"reasoning,omitempty"`
	Text            *textConfig    `json:"text,omitempty"`
	PromptCacheKey  string         `json:"prompt_cache_key,omitempty"`
	Store           bool           `json:"store"`
	Include         []string       `json:"include,omitempty"`
	Stream          bool           `json:"stream,omitempty"`
	extra           map[string]any `json:"-"`
}

// MarshalJSON merges the extra fields over the typed ones, which they
// override, the way the other providers merge ExtraFields.
func (r request) MarshalJSON() ([]byte, error) {
	type plain request
	raw, err := json.Marshal(plain(r))
	if err != nil || len(r.extra) == 0 {
		return raw, err
	}
	var merged map[string]any
	if err := json.Unmarshal(raw, &merged); err != nil {
		return nil, err
	}
	for k, v := range r.extra {
		merged[k] = v
	}
	return json.Marshal(merged)
}

type tool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict"`
}

type reasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type textConfig struct {
	Format textFormat `json:"format"`
}

type textFormat struct {
	Type        string `json:"type"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}

// includeEncryptedReasoning asks for the reasoning items to come back with
// their encrypted content. Requests are not stored (store: false), so it is
// the only way to replay a model's reasoning on the next turn.
const includeEncryptedReasoning = "reasoning.encrypted_content"

// thresholds of a reasoning budget, in tokens, used to pick an effort level
// when the caller gives a budget and no max_output_tokens to relate it to.
const (
	lowBudget    = 2048
	mediumBudget = 8192
)

// buildRequest converts genai chat completion options to a Responses API
// request.
func buildRequest(opts *llm.ChatCompletionOptions, model string, stream bool) (*request, error) {
	if opts.Audio != nil {
		return nil, llm.NewValidationError("audio", "the Responses API has no audio output")
	}
	for _, modality := range opts.Modalities {
		if modality != "text" {
			return nil, llm.NewValidationError("modalities", fmt.Sprintf("the Responses API has no %q output", modality))
		}
	}

	input, err := buildInput(opts.Messages)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	req := &request{
		Model:           model,
		Input:           input,
		Temperature:     opts.Temperature,
		MaxOutputTokens: opts.MaxCompletionTokens,
		PromptCacheKey:  opts.SessionID,
		Store:           false,
		Include:         []string{includeEncryptedReasoning},
		Stream:          stream,
		extra:           opts.ExtraFields,
	}

	if len(opts.Tools) > 0 {
		req.Tools = make([]tool, 0, len(opts.Tools))
		for _, t := range opts.Tools {
			parameters := t.Parameters()
			if parameters == nil {
				parameters = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			// Not strict: strict mode demands every property be required
			// and additionalProperties be false, which genai's tool schemas
			// do not promise.
			req.Tools = append(req.Tools, tool{
				Type:        "function",
				Name:        t.Name(),
				Description: t.Description(),
				Parameters:  parameters,
			})
		}
		req.ToolChoice = string(opts.ToolChoice)
	}

	req.Reasoning = buildReasoning(opts.Reasoning, opts.MaxCompletionTokens)

	if opts.ResponseFormat == llm.ResponseFormatJSON {
		format := textFormat{Type: "json_object"}
		if opts.ResponseSchema != nil {
			strict := llm.IsStrictResponseSchema(opts.ResponseSchema)
			format = textFormat{
				Type:        "json_schema",
				Name:        opts.ResponseSchema.Name(),
				Description: opts.ResponseSchema.Description(),
				Schema:      opts.ResponseSchema.Schema(),
				Strict:      &strict,
			}
		}
		req.Text = &textConfig{Format: format}
	}

	return req, nil
}

// buildReasoning maps llm.ReasoningOptions to the reasoning configuration.
// The API only takes an effort level: a token budget is converted to the
// level whose share of max_output_tokens it is closest to, or by absolute
// thresholds when no max_output_tokens is set.
func buildReasoning(opts *llm.ReasoningOptions, maxOutputTokens *int) *reasoning {
	if opts == nil {
		return nil
	}

	var effort llm.ReasoningEffort
	switch {
	case opts.Enabled != nil && !*opts.Enabled:
		effort = llm.ReasoningEffortNone
	case opts.Effort != nil:
		effort = *opts.Effort
	case opts.MaxTokens != nil:
		effort = effortForBudget(*opts.MaxTokens, maxOutputTokens)
	case opts.Enabled != nil && *opts.Enabled:
		effort = llm.ReasoningEffortMedium
	default:
		return nil
	}

	r := &reasoning{Effort: string(effort)}
	// A summary is the only readable reasoning the API returns. A disabled
	// reasoning has none to summarize.
	if !opts.Exclude && effort != llm.ReasoningEffortNone {
		r.Summary = "auto"
	}
	return r
}

// effortRatios is the share of max_tokens each effort stands for, as
// documented on llm.ReasoningEffort.
var effortRatios = []struct {
	effort llm.ReasoningEffort
	ratio  float64
}{
	{llm.ReasoningEffortMinimal, 0.10},
	{llm.ReasoningEffortLow, 0.20},
	{llm.ReasoningEffortMedium, 0.50},
	{llm.ReasoningEffortHigh, 0.80},
	{llm.ReasoningEffortXHigh, 0.95},
}

func effortForBudget(budget int, maxOutputTokens *int) llm.ReasoningEffort {
	if maxOutputTokens == nil || *maxOutputTokens <= 0 {
		switch {
		case budget <= lowBudget:
			return llm.ReasoningEffortLow
		case budget <= mediumBudget:
			return llm.ReasoningEffortMedium
		default:
			return llm.ReasoningEffortHigh
		}
	}

	ratio := float64(budget) / float64(*maxOutputTokens)
	closest := effortRatios[0]
	for _, candidate := range effortRatios[1:] {
		if abs(candidate.ratio-ratio) < abs(closest.ratio-ratio) {
			closest = candidate
		}
	}
	return closest.effort
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

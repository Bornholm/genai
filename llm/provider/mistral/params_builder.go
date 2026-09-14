package mistral

import (
	"context"
	"log/slog"
	"slices"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/pkg/errors"

	genai "github.com/bornholm/genai/llm/provider/openai"
)

type paramsBuilder struct {
	model string
}

func (b *paramsBuilder) BuildParams(ctx context.Context, opts *llm.ChatCompletionOptions) (*openai.ChatCompletionNewParams, error) {
	if b.model == "" {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	params, err := genai.ConfigureParams(
		ctx, opts,
		genai.ConfigureTools,
		genai.ConfigureTemperature,
		genai.ConfigureResponseFormat,
		ConfigureMistralMessages,
		configureMistralMaxTokens,
		configureRandomSeed,
		genai.ConfigureReasoning,
		configurePromptMode,
		configureMistralExtraFields,
	)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	params.Model = openai.ChatModel(b.model)

	return params, nil
}

var _ genai.ParamsBuilder = &paramsBuilder{}

// configureMistralMaxTokens maps the internal MaxCompletionTokens option to the
// Mistral-specific "max_tokens" field. Mistral does not accept "max_completion_tokens"
// (the OpenAI o1 field name) and returns a 422 error when it is present.
func configureMistralMaxTokens(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
	if opts.MaxCompletionTokens == nil {
		return nil
	}
	genai.MergeExtraFields(params, map[string]any{
		"max_tokens": *opts.MaxCompletionTokens,
	})
	return nil
}

func configureRandomSeed(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
	if opts.Seed == nil {
		return nil
	}

	genai.MergeExtraFields(params, map[string]any{
		"random_seed": *opts.Seed,
	})
	return nil
}

// promptMode represents the Mistral prompt_mode parameter
type promptMode string

const (
	promptModeReasoning promptMode = "reasoning"
)

// configurePromptMode adds the prompt_mode parameter for Mistral reasoning models.
// This controls whether the default reasoning system prompt is used.
// By default, reasoning models use the "reasoning" prompt_mode.
// Set prompt_mode to "null" to opt out of the default system prompt.
func configurePromptMode(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
	// Check if reasoning options are set - if so, use reasoning prompt_mode
	// If users want to opt out, they would need to set extra_fields manually
	// For now, we default to "reasoning" for reasoning models
	if opts.Reasoning != nil {
		genai.MergeExtraFields(params, map[string]any{
			"prompt_mode": string(promptModeReasoning),
		})
	}

	return nil
}

// openAIPlatformFields lists request fields that belong to the OpenAI platform
// rather than to inference itself — response storage, routing, telemetry. The
// proxy forwards client fields it has no mapping for verbatim, so that portable
// sampling parameters are not silently dropped, but Mistral validates its
// request body strictly and answers 422 "Extra inputs are not permitted" on any
// of these. Clients built against the OpenAI API send them without knowing the
// request will be routed elsewhere.
var openAIPlatformFields = map[string]struct{}{
	"store":             {},
	"metadata":          {},
	"service_tier":      {},
	"prompt_cache_key":  {},
	"safety_identifier": {},
	"logit_bias":        {},
	"logprobs":          {},
	"top_logprobs":      {},
	"modalities":        {},
	"audio":             {},
}

// configureMistralExtraFields injects the caller-provided extra fields the way
// genai.ConfigureExtraFields does, minus the OpenAI-only ones Mistral rejects.
// Like its generic counterpart it merges with the fields earlier configurators
// set, so it must run last in the chain.
func configureMistralExtraFields(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
	if len(opts.ExtraFields) == 0 {
		return nil
	}

	filtered := make(map[string]any, len(opts.ExtraFields))
	var dropped []string
	for k, v := range opts.ExtraFields {
		if _, unsupported := openAIPlatformFields[k]; unsupported {
			dropped = append(dropped, k)
			continue
		}
		filtered[k] = v
	}

	if len(dropped) > 0 {
		slices.Sort(dropped)
		slog.DebugContext(ctx, "dropped request fields unsupported by mistral", slog.Any("fields", dropped))
	}

	genai.MergeExtraFields(params, filtered)
	return nil
}

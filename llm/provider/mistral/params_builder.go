package mistral

import (
	"context"

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
		genai.ConfigureMaxCompletionTokens,
		configureRandomSeed,
		genai.ConfigureReasoning,
		configurePromptMode,
	)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	params.Model = openai.ChatModel(b.model)

	return params, nil
}

var _ genai.ParamsBuilder = &paramsBuilder{}

func configureRandomSeed(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
	if opts.Seed == nil {
		return nil
	}

	params.WithExtraFields(map[string]any{
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
		params.WithExtraFields(map[string]any{
			"prompt_mode": string(promptModeReasoning),
		})
	}

	return nil
}

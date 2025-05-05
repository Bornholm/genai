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
		genai.ConfigureMessages,
		genai.ConfigureMaxCompletionTokens,
		configureRandomSeed,
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

package minimax

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

const Name provider.Name = "minimax"

func init() {
	provider.RegisterImageGeneration(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.ImageGenerationClient, error) {
			return NewImageGenerationClient(nil, opts.BaseURL, opts.APIKey, opts.Model), nil
		},
	)
}

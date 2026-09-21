package typesafe

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

const Name provider.Name = "typesafe"

func init() {
	provider.RegisterDecision(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.DecisionClient, error) {
			return NewDecisionClientForBaseURL(opts.BaseURL, opts.APIKey, opts.Model), nil
		},
	)
}

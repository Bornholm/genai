package openai

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/pkg/errors"
)

const Name provider.Name = "openai"

func init() {
	provider.Register(Name, func(ctx context.Context) (llm.Client, error) {
		baseURL, err := provider.ContextAPIBaseURL(ctx)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		apiKey, err := provider.ContextAPIKey(ctx)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		model, err := provider.ContextModel(ctx)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		client := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithAPIKey(apiKey),
		)

		return NewClient(client, model), nil
	})
}

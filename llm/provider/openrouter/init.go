package openrouter

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/pkg/errors"
	"github.com/revrost/go-openrouter"
)

const Name provider.Name = "openrouter"

func init() {
	provider.Register(Name, func(ctx context.Context) (llm.Client, error) {
		model, err := provider.ContextModel(ctx)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		apiKey, err := provider.ContextKey(ctx)
		if err != nil && !errors.Is(err, provider.ErrContextKeyNotFound) {
			return nil, errors.WithStack(err)
		}

		client := openrouter.NewClient(apiKey)

		return NewClient(client, model), nil
	})
}

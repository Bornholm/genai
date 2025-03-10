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
		baseURL, err := provider.ContextBaseURL(ctx)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		options := []option.RequestOption{
			option.WithBaseURL(baseURL),
		}

		chatCompletionModel, err := provider.ContextChatCompletionModel(ctx)
		if err != nil && !errors.Is(err, provider.ErrContextKeyNotFound) {
			return nil, errors.WithStack(err)
		}

		embeddingsModel, err := provider.ContextEmbeddingsModel(ctx)
		if err != nil && !errors.Is(err, provider.ErrContextKeyNotFound) {
			return nil, errors.WithStack(err)
		}

		key, err := provider.ContextKey(ctx)
		if err != nil && !errors.Is(err, provider.ErrContextKeyNotFound) {
			return nil, errors.WithStack(err)
		}

		if key != "" {
			options = append(options, option.WithAPIKey(key))
		}

		client := openai.NewClient(
			options...,
		)

		return NewClient(client, chatCompletionModel, embeddingsModel), nil
	})
}

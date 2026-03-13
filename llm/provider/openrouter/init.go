package openrouter

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/revrost/go-openrouter"
)

const Name provider.Name = "openrouter"

func init() {
	provider.RegisterChatCompletion(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.ChatCompletionClient, error) {
			client := openrouter.NewClient(opts.APIKey)
			return NewChatCompletionClient(client, opts.Model), nil
		},
	)

	provider.RegisterEmbeddings(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.EmbeddingsClient, error) {
			client := openrouter.NewClient(opts.APIKey)
			return NewEmbeddingsClient(client, opts.Model), nil
		},
	)
}

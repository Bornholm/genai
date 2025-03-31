package openrouter

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/revrost/go-openrouter"
)

const Name provider.Name = "openrouter"

func init() {
	provider.RegisterChatCompletion(Name, func(ctx context.Context, opts provider.ClientOptions) (llm.ChatCompletionClient, error) {
		client := openrouter.NewClient(opts.APIKey)
		return NewChatCompletionClient(client, opts.Model), nil
	})
}

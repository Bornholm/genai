package openai

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const Name provider.Name = "openai"

func init() {
	provider.RegisterChatCompletion(Name, func(ctx context.Context, opts provider.ClientOptions) (llm.ChatCompletionClient, error) {
		options := []option.RequestOption{
			option.WithBaseURL(opts.BaseURL),
		}

		if opts.APIKey != "" {
			options = append(options, option.WithAPIKey(opts.APIKey))
		}

		client := openai.NewClient(
			options...,
		)

		return NewChatCompletionClient(client, opts.Model), nil
	})

	provider.RegisterEmbeddings(Name, func(ctx context.Context, opts provider.ClientOptions) (llm.EmbeddingsClient, error) {
		options := []option.RequestOption{
			option.WithBaseURL(opts.BaseURL),
		}

		if opts.APIKey != "" {
			options = append(options, option.WithAPIKey(opts.APIKey))
		}

		client := openai.NewClient(
			options...,
		)

		return NewEmbeddingsClient(client, opts.Model), nil
	})
}

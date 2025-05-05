package mistral

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

	genai "github.com/bornholm/genai/llm/provider/openai"
)

const Name provider.Name = "mistral"

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

		return genai.NewChatCompletionClient(client, &paramsBuilder{
			model: opts.Model,
		}), nil
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

		return genai.NewEmbeddingsClient(client, opts.Model), nil
	})
}

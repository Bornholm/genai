package mistral

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	genai "github.com/bornholm/genai/llm/provider/openai"
	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const Name provider.Name = "mistral"

func init() {
	provider.RegisterChatCompletion(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.ChatCompletionClient, error) {
			options := []option.RequestOption{
				option.WithBaseURL(opts.BaseURL),
			}
			if opts.APIKey != "" {
				options = append(options, option.WithAPIKey(opts.APIKey))
			}
			client := openaisdk.NewClient(options...)
			return NewChatCompletionClient(client, &paramsBuilder{model: opts.Model}), nil
		},
	)

	provider.RegisterEmbeddings(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.EmbeddingsClient, error) {
			options := []option.RequestOption{
				option.WithBaseURL(opts.BaseURL),
			}
			if opts.APIKey != "" {
				options = append(options, option.WithAPIKey(opts.APIKey))
			}
			client := openaisdk.NewClient(options...)
			return genai.NewEmbeddingsClient(client, opts.Model), nil
		},
	)
}

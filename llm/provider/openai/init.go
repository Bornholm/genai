package openai

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const Name provider.Name = "openai"

func init() {
	provider.RegisterChatCompletion(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.ChatCompletionClient, error) {
			client := NewSDKClient(opts.CommonOptions)
			return NewChatCompletionClient(client, &paramsBuilder{model: opts.Model}), nil
		},
	)

	provider.RegisterEmbeddings(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.EmbeddingsClient, error) {
			client := NewSDKClient(opts.CommonOptions)
			return NewEmbeddingsClient(client, opts.Model), nil
		},
	)

	provider.RegisterTranscription(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.TranscriptionClient, error) {
			client := NewSDKClient(opts.CommonOptions)
			return NewTranscriptionClient(client, opts.Model), nil
		},
	)

	provider.RegisterImageGeneration(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.ImageGenerationClient, error) {
			client := NewSDKClient(opts.CommonOptions)
			return NewImageGenerationClient(client, opts.Model), nil
		},
	)

	provider.RegisterDiscovery(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.DiscoveryClient, error) {
			client := NewSDKClient(opts.CommonOptions)
			return NewDiscoveryClient(client), nil
		},
	)
}

// NewSDKClient builds the OpenAI SDK client for an OpenAI-compatible API:
// its base URL and key, and the headers configured on top of the SDK's.
func NewSDKClient(opts provider.CommonOptions) openaisdk.Client {
	options := []option.RequestOption{
		option.WithBaseURL(opts.BaseURL),
		option.WithMaxRetries(0), // genai's llmretry wrapper handles all retries
		option.WithMiddleware(skipEmptySSEEvents),
	}
	if opts.APIKey != "" {
		options = append(options, option.WithAPIKey(opts.APIKey))
	}
	if client := opts.HTTPClient(nil); client != nil {
		options = append(options, option.WithHTTPClient(client))
	}
	return openaisdk.NewClient(options...)
}

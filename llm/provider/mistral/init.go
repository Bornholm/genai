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
				option.WithMaxRetries(0), // genai's llmretry wrapper handles all retries
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
				option.WithMaxRetries(0), // genai's llmretry wrapper handles all retries
			}
			if opts.APIKey != "" {
				options = append(options, option.WithAPIKey(opts.APIKey))
			}
			client := openaisdk.NewClient(options...)
			return genai.NewEmbeddingsClient(client, opts.Model), nil
		},
	)

	// L'API de transcription Mistral (Voxtral) est compatible avec l'endpoint
	// multipart /audio/transcriptions d'OpenAI : on réutilise le client openai.
	provider.RegisterTranscription(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.TranscriptionClient, error) {
			options := []option.RequestOption{
				option.WithBaseURL(opts.BaseURL),
				option.WithMaxRetries(0), // genai's llmretry wrapper handles all retries
			}
			if opts.APIKey != "" {
				options = append(options, option.WithAPIKey(opts.APIKey))
			}
			client := openaisdk.NewClient(options...)
			return genai.NewTranscriptionClient(client, opts.Model), nil
		},
	)
}

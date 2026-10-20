package mistral

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	genai "github.com/bornholm/genai/llm/provider/openai"
)

const Name provider.Name = "mistral"

func init() {
	provider.RegisterChatCompletion(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.ChatCompletionClient, error) {
			client := genai.NewSDKClient(opts.CommonOptions)
			return NewChatCompletionClient(client, &paramsBuilder{model: opts.Model}), nil
		},
	)

	provider.RegisterEmbeddings(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.EmbeddingsClient, error) {
			client := genai.NewSDKClient(opts.CommonOptions)
			return genai.NewEmbeddingsClient(client, opts.Model), nil
		},
	)

	// L'API de transcription Mistral (Voxtral) est compatible avec l'endpoint
	// multipart /audio/transcriptions d'OpenAI : on réutilise le client openai.
	provider.RegisterTranscription(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.TranscriptionClient, error) {
			client := genai.NewSDKClient(opts.CommonOptions)
			return genai.NewTranscriptionClient(client, opts.Model), nil
		},
	)

	provider.RegisterDiscovery(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.DiscoveryClient, error) {
			client := genai.NewSDKClient(opts.CommonOptions)
			return genai.NewDiscoveryClient(client), nil
		},
	)
}

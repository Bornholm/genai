package openrouter_test

import (
	"context"
	"os"
	"testing"

	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider"
	openrouterProvider "github.com/bornholm/genai/llm/provider/openrouter"
)

func TestConformance(t *testing.T) {
	apiKey := os.Getenv("CONFORMANCE_OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("CONFORMANCE_OPENROUTER_API_KEY not set")
	}

	chatModel := os.Getenv("CONFORMANCE_OPENROUTER_CHAT_MODEL")
	if chatModel == "" {
		chatModel = "openai/gpt-oss-20b:free"
	}

	transcriptionModel := os.Getenv("CONFORMANCE_OPENROUTER_TRANSCRIPTION_MODEL")
	if transcriptionModel == "" {
		transcriptionModel = "openai/whisper-1"
	}

	ctx := context.Background()
	client, err := provider.Create(ctx,
		func(opts *provider.Options) error {
			opts.ChatCompletion = &provider.ResolvedClientOptions{
				Provider: openrouterProvider.Name,
				Specific: &openrouterProvider.Options{
					CommonOptions: provider.CommonOptions{
						APIKey: apiKey,
						Model:  chatModel,
					},
				},
			}
			opts.Transcription = &provider.ResolvedClientOptions{
				Provider: openrouterProvider.Name,
				Specific: &openrouterProvider.Options{
					CommonOptions: provider.CommonOptions{
						APIKey: apiKey,
						Model:  transcriptionModel,
					},
				},
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	conformance.New(client,
		conformance.WithFeatures(
			conformance.FeatureChatCompletion|
				conformance.FeatureStreaming|
				conformance.FeatureToolCalls|
				conformance.FeatureJSON|
				conformance.FeatureMultimodal|
				conformance.FeatureTranscription,
		),
	).Run(t)
}

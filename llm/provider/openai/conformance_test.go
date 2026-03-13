package openai_test

import (
	"context"
	"os"
	"testing"

	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider"
	openaiProvider "github.com/bornholm/genai/llm/provider/openai"
)

func TestConformance(t *testing.T) {
	apiKey := os.Getenv("CONFORMANCE_OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("CONFORMANCE_OPENAI_API_KEY not set")
	}

	chatModel := os.Getenv("CONFORMANCE_OPENAI_CHAT_MODEL")
	if chatModel == "" {
		chatModel = "gpt-4o-mini"
	}

	embeddingModel := os.Getenv("CONFORMANCE_OPENAI_EMBEDDING_MODEL")
	if embeddingModel == "" {
		embeddingModel = "text-embedding-3-small"
	}

	ctx := context.Background()
	client, err := provider.Create(ctx,
		func(opts *provider.Options) error {
			opts.ChatCompletion = &provider.ResolvedClientOptions{
				Provider: openaiProvider.Name,
				Specific: &openaiProvider.Options{
					CommonOptions: provider.CommonOptions{
						BaseURL: "https://api.openai.com/v1",
						APIKey:  apiKey,
						Model:   chatModel,
					},
				},
			}
			opts.Embeddings = &provider.ResolvedClientOptions{
				Provider: openaiProvider.Name,
				Specific: &openaiProvider.Options{
					CommonOptions: provider.CommonOptions{
						BaseURL: "https://api.openai.com/v1",
						APIKey:  apiKey,
						Model:   embeddingModel,
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
				conformance.FeatureEmbeddings,
		),
	).Run(t)
}

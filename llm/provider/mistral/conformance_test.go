package mistral_test

import (
	"context"
	"os"
	"testing"

	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider"
	mistralProvider "github.com/bornholm/genai/llm/provider/mistral"
)

func TestConformance(t *testing.T) {
	apiKey := os.Getenv("CONFORMANCE_MISTRAL_API_KEY")
	if apiKey == "" {
		t.Skip("CONFORMANCE_MISTRAL_API_KEY not set")
	}

	chatModel := os.Getenv("CONFORMANCE_MISTRAL_CHAT_MODEL")
	if chatModel == "" {
		chatModel = "magistral-small-latest"
	}

	embeddingModel := os.Getenv("CONFORMANCE_MISTRAL_EMBEDDING_MODEL")
	if embeddingModel == "" {
		embeddingModel = "mistral-embed"
	}

	ctx := context.Background()
	client, err := provider.Create(ctx,
		func(opts *provider.Options) error {
			opts.ChatCompletion = &provider.ResolvedClientOptions{
				Provider: mistralProvider.Name,
				Specific: &mistralProvider.Options{
					CommonOptions: provider.CommonOptions{
						BaseURL: "https://api.mistral.ai/v1",
						APIKey:  apiKey,
						Model:   chatModel,
					},
				},
			}
			opts.Embeddings = &provider.ResolvedClientOptions{
				Provider: mistralProvider.Name,
				Specific: &mistralProvider.Options{
					CommonOptions: provider.CommonOptions{
						BaseURL: "https://api.mistral.ai/v1",
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

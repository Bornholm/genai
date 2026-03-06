package mistral_test

import (
	"context"
	"os"
	"testing"

	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider"
	_ "github.com/bornholm/genai/llm/provider/mistral"
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
		provider.WithChatCompletionOptions(provider.ClientOptions{
			Provider: "mistral",
			BaseURL:  "https://api.mistral.ai/v1",
			APIKey:   apiKey,
			Model:    chatModel,
		}),
		provider.WithEmbeddingsOptions(provider.ClientOptions{
			Provider: "mistral",
			BaseURL:  "https://api.mistral.ai/v1",
			APIKey:   apiKey,
			Model:    embeddingModel,
		}),
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

package openrouter_test

import (
	"context"
	"os"
	"testing"

	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider"
	_ "github.com/bornholm/genai/llm/provider/openrouter"
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

	ctx := context.Background()
	client, err := provider.Create(ctx,
		provider.WithChatCompletionOptions(provider.ClientOptions{
			Provider: "openrouter",
			BaseURL:  "https://openrouter.ai/api/v1",
			APIKey:   apiKey,
			Model:    chatModel,
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
				conformance.FeatureMultimodal,
		),
	).Run(t)
}

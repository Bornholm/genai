package anthropic_test

import (
	"context"
	"os"
	"testing"

	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider"
	anthropicProvider "github.com/bornholm/genai/llm/provider/anthropic"
)

func TestConformance(t *testing.T) {
	apiKey := os.Getenv("CONFORMANCE_ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("CONFORMANCE_ANTHROPIC_API_KEY not set")
	}

	chatModel := os.Getenv("CONFORMANCE_ANTHROPIC_CHAT_MODEL")
	if chatModel == "" {
		chatModel = "claude-haiku-4-5"
	}

	ctx := context.Background()
	client, err := provider.Create(ctx,
		func(opts *provider.Options) error {
			opts.ChatCompletion = &provider.ResolvedClientOptions{
				Provider: anthropicProvider.Name,
				Specific: &anthropicProvider.Options{
					CommonOptions: provider.CommonOptions{
						APIKey: apiKey,
						Model:  chatModel,
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
				conformance.FeatureReasoning,
		),
	).Run(t)
}

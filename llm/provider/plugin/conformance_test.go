package plugin

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider"
)

// TestConformance runs the conformance suite against the openai provider
// wrapped in a plugin, so that every feature crosses the process boundary.
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

	binary := filepath.Join(t.TempDir(), "genai-provider-openai-plugin")
	build := exec.Command("go", "build", "-o", binary, "./openaiplugin")
	build.Dir = filepath.Join("..", "..", "..", "plugin", "testplugins")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		t.Fatalf("could not build openai plugin: %v", err)
	}

	ctx := context.Background()
	chat, err := NewChatCompletionClient(ctx, binary, map[string]string{"API_KEY": apiKey, "MODEL": chatModel})
	if err != nil {
		t.Fatalf("could not configure chat client: %+v", err)
	}
	embeddings, err := NewEmbeddingsClient(ctx, binary, map[string]string{"API_KEY": apiKey, "MODEL": embeddingModel})
	if err != nil {
		t.Fatalf("could not configure embeddings client: %+v", err)
	}
	defer chat.Process().Kill()

	conformance.New(provider.NewClient(chat, embeddings, nil),
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

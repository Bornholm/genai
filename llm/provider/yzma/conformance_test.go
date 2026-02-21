package yzma_test

import (
	"os"
	"testing"

	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider/yzma"
)

func TestConformance(t *testing.T) {
	modelPath := os.Getenv("CONFORMANCE_YZMA_MODEL_PATH")
	modelURL := os.Getenv("CONFORMANCE_YZMA_MODEL_URL")
	if modelPath == "" && modelURL == "" {
		t.Skip("CONFORMANCE_YZMA_MODEL_PATH or CONFORMANCE_YZMA_MODEL_URL not set")
	}

	libPath := os.Getenv("CONFORMANCE_YZMA_LIB_PATH")

	opts := []yzma.OptionFunc{}
	if modelPath != "" {
		opts = append(opts, yzma.WithModelPath(modelPath))
	}
	if modelURL != "" {
		opts = append(opts, yzma.WithModelURL(modelURL))
	}
	if libPath != "" {
		opts = append(opts, yzma.WithLibPath(libPath))
	}

	client, err := yzma.NewChatCompletionClient(opts...)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	conformance.New(client,
		conformance.WithFeatures(
			conformance.FeatureChatCompletion|
				conformance.FeatureStreaming|
				conformance.FeatureToolCalls|
				conformance.FeatureJSON,
		),
	).Run(t)
}

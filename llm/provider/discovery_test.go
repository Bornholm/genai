package provider_test

import (
	"context"
	"testing"

	"github.com/pkg/errors"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

type stubDiscovery struct{}

func (stubDiscovery) Models(context.Context) ([]llm.ModelInfo, error) {
	return []llm.ModelInfo{llm.NewModelInfo("m")}, nil
}

func TestCreate_DiscoveryAloneIsAClient(t *testing.T) {
	const name provider.Name = "test-discovery-only"
	provider.RegisterDiscovery(name,
		func() *testEmbeddingsOptions { return &testEmbeddingsOptions{} },
		func(context.Context, *testEmbeddingsOptions) (llm.DiscoveryClient, error) {
			return stubDiscovery{}, nil
		},
	)

	client, err := provider.Create(context.Background(), provider.WithDiscovery(name, testEmbeddingsOptions{}))
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	models, err := client.(llm.DiscoveryClient).Models(context.Background())
	if err != nil || len(models) != 1 || models[0].ID() != "m" {
		t.Fatalf("models = %v, %v", models, err)
	}

	if _, err := client.ChatCompletion(context.Background()); !errors.Is(err, llm.ErrUnavailable) {
		t.Errorf("chat completion without a provider must be unavailable, got %v", err)
	}
}

func TestClient_ModelsUnavailableWithoutDiscovery(t *testing.T) {
	client := provider.NewClient(nil, nil, nil)
	if _, err := client.Models(context.Background()); !errors.Is(err, llm.ErrUnavailable) {
		t.Errorf("err = %v, want ErrUnavailable", err)
	}
}

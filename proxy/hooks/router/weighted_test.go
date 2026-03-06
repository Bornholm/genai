package router

import (
	"context"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/proxy"
)

// testClient satisfies llm.Client for testing only.
type testClient struct{ id string }

func (m *testClient) ChatCompletion(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return nil, nil
}
func (m *testClient) ChatCompletionStream(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	return nil, nil
}
func (m *testClient) Embeddings(_ context.Context, _ []string, _ ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

var _ llm.Client = &testClient{}

func TestWeightedRouter_SingleBackend(t *testing.T) {
	c := &testClient{id: "c1"}
	r := NewWeightedRouter(map[string][]WeightedBackend{
		"gpt-4": {{Client: c, Weight: 1, Model: "real-model"}},
	}, 0)

	req := &proxy.ProxyRequest{Model: "gpt-4"}
	client, model, err := r.ResolveModel(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "real-model" {
		t.Errorf("model = %q, want real-model", model)
	}
	if client != llm.Client(c) {
		t.Error("wrong client returned")
	}
}

func TestWeightedRouter_ModelNotFound(t *testing.T) {
	r := NewWeightedRouter(nil, 0)
	req := &proxy.ProxyRequest{Model: "unknown"}

	_, _, err := r.ResolveModel(context.Background(), req)
	if err != proxy.ErrModelNotFound {
		t.Errorf("expected ErrModelNotFound, got %v", err)
	}
}

func TestWeightedRouter_WeightedDistribution(t *testing.T) {
	c1 := &testClient{id: "c1"}
	c2 := &testClient{id: "c2"}

	r := NewWeightedRouter(map[string][]WeightedBackend{
		"gpt-4": {
			{Client: c1, Weight: 9, Model: "m1"},
			{Client: c2, Weight: 1, Model: "m2"},
		},
	}, 0)

	counts := map[string]int{}
	n := 1000
	req := &proxy.ProxyRequest{Model: "gpt-4"}

	for i := 0; i < n; i++ {
		client, _, err := r.ResolveModel(context.Background(), req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// identify client by comparing to known instances
		switch client {
		case llm.Client(c1):
			counts["c1"]++
		case llm.Client(c2):
			counts["c2"]++
		}
	}

	// With weight 9:1, c1 should get ~90% of requests. Allow ±15%.
	ratio := float64(counts["c1"]) / float64(n)
	if ratio < 0.75 || ratio > 0.99 {
		t.Errorf("c1 ratio = %.2f, expected ~0.90", ratio)
	}
}

func TestWeightedRouter_ListModels(t *testing.T) {
	r := NewWeightedRouter(map[string][]WeightedBackend{
		"gpt-4":   {{Client: &testClient{}, Weight: 1}},
		"gpt-3.5": {{Client: &testClient{}, Weight: 1}},
	}, 0)

	models, err := r.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 2 {
		t.Errorf("models = %d, want 2", len(models))
	}
}

func TestWeightedRouter_AddBackend(t *testing.T) {
	r := NewWeightedRouter(nil, 0)
	c := &testClient{id: "dynamic"}
	r.AddBackend("gpt-5", WeightedBackend{Client: c, Weight: 1, Model: "gpt-5-real"})

	req := &proxy.ProxyRequest{Model: "gpt-5"}
	client, _, err := r.ResolveModel(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client != llm.Client(c) {
		t.Error("wrong client")
	}
}

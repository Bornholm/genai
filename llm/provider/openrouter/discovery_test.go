package openrouter

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

func TestDiscovery_ReadsTheLimitsOfEachModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/models" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[
			{"id":"z-ai/glm-5.3","name":"Z.ai: GLM 5.3","created":1760000000,"context_length":1000000,
				"top_provider":{"context_length":1000000,"max_completion_tokens":131072,"is_moderated":false}},
			{"id":"openrouter/auto","name":"Auto Router","created":0,"top_provider":{"is_moderated":false}}
		]}`))
	}))
	defer server.Close()

	models, err := NewDiscoveryClient(nil, server.URL+"/api/v1", "").Models(context.Background())
	if err != nil {
		t.Fatalf("models: %v", err)
	}

	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	glm := models[0]
	if glm.ID() != "z-ai/glm-5.3" || glm.DisplayName() != "Z.ai: GLM 5.3" ||
		glm.ContextWindow() != 1000000 || glm.MaxOutputTokens() != 131072 || glm.CreatedAt().IsZero() {
		t.Errorf("glm = %q %q ctx=%d out=%d at %v", glm.ID(), glm.DisplayName(), glm.ContextWindow(), glm.MaxOutputTokens(), glm.CreatedAt())
	}
	auto := models[1]
	if auto.ContextWindow() != 0 || auto.MaxOutputTokens() != 0 || !auto.CreatedAt().IsZero() {
		t.Errorf("missing limits must stay zero, got ctx=%d out=%d at %v", auto.ContextWindow(), auto.MaxOutputTokens(), auto.CreatedAt())
	}
}

func TestDiscovery_RegisteredFactoryUsesTheConfiguredAccount(t *testing.T) {
	var path, auth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path, auth = r.URL.Path, r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"z-ai/glm-5.3","name":"GLM 5.3","top_provider":{}}]}`))
	}))
	defer server.Close()

	// A trailing slash on the base URL must not double the one of the route.
	client, err := provider.Create(context.Background(), provider.WithDiscovery(Name, Options{
		CommonOptions: provider.CommonOptions{BaseURL: server.URL + "/api/v1/", APIKey: "sk-or"},
	}))
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	models, err := client.(llm.DiscoveryClient).Models(context.Background())
	if err != nil {
		t.Fatalf("models: %v", err)
	}

	if path != "/api/v1/models" {
		t.Errorf("path = %q, want /api/v1/models", path)
	}
	if auth != "Bearer sk-or" {
		t.Errorf("authorization = %q, want the API key", auth)
	}
	if len(models) != 1 || models[0].ID() != "z-ai/glm-5.3" {
		t.Errorf("models = %v", models)
	}
}

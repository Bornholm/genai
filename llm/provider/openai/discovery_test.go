package openai_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/openai"
)

func TestDiscovery_ListsTheModelsOfAnOpenAICompatibleGateway(t *testing.T) {
	var auth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/inference/go/openai/v1/models" {
			http.NotFound(w, r)
			return
		}
		auth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[
			{"id":"glm-5.3","object":"model","created":1760000000,"owned_by":"zai"},
			{"id":"kimi-k3","object":"model","created":0,"owned_by":"moonshot"}
		]}`))
	}))
	defer server.Close()

	client, err := provider.Create(context.Background(), provider.WithDiscovery(openai.Name, openai.Options{
		CommonOptions: provider.CommonOptions{BaseURL: server.URL + "/inference/go/openai/v1", APIKey: "sk-test"},
	}))
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	models, err := client.(llm.DiscoveryClient).Models(context.Background())
	if err != nil {
		t.Fatalf("models: %v", err)
	}

	if auth != "Bearer sk-test" {
		t.Errorf("authorization = %q, want the API key", auth)
	}
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	if models[0].ID() != "glm-5.3" || !models[0].CreatedAt().Equal(time.Unix(1760000000, 0)) {
		t.Errorf("first model = %q created %v", models[0].ID(), models[0].CreatedAt())
	}
	if models[1].ID() != "kimi-k3" || !models[1].CreatedAt().IsZero() {
		t.Errorf("an unknown creation date must stay zero, got %v", models[1].CreatedAt())
	}
}

func TestDiscovery_ReportsAnUpstreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid api key","type":"invalid_request_error"}}`))
	}))
	defer server.Close()

	client, err := provider.Create(context.Background(), provider.WithDiscovery(openai.Name, openai.Options{
		CommonOptions: provider.CommonOptions{BaseURL: server.URL, APIKey: "sk-wrong"},
	}))
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	if _, err := client.(llm.DiscoveryClient).Models(context.Background()); err == nil {
		t.Fatal("a 401 must surface as an error")
	}
}

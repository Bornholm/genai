package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

func TestDiscovery_FollowsThePagination(t *testing.T) {
	var cursors []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("X-Api-Key") != "sk-ant" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		cursors = append(cursors, r.URL.Query().Get("after_id"))
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Query().Get("after_id") {
		case "":
			_, _ = w.Write([]byte(`{"data":[{"id":"qwen3.8-max","type":"model","display_name":"Qwen 3.8 Max",
				"created_at":"2026-09-01T00:00:00Z","max_input_tokens":1000000,"max_tokens":131072}],
				"has_more":true,"first_id":"qwen3.8-max","last_id":"qwen3.8-max"}`))
		case "qwen3.8-max":
			_, _ = w.Write([]byte(`{"data":[{"id":"minimax-m3","type":"model","display_name":"MiniMax M3",
				"created_at":"1970-01-01T00:00:00Z"}],
				"has_more":false,"first_id":"minimax-m3","last_id":"minimax-m3"}`))
		default:
			w.WriteHeader(http.StatusBadRequest)
		}
	}))
	defer server.Close()

	// The trailing /v1 is stripped as for chat completion.
	client, err := provider.Create(context.Background(), provider.WithDiscovery(Name, Options{
		CommonOptions: provider.CommonOptions{BaseURL: server.URL + "/v1", APIKey: "sk-ant"},
	}))
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	models, err := client.(llm.DiscoveryClient).Models(context.Background())
	if err != nil {
		t.Fatalf("models: %v", err)
	}

	if len(cursors) != 2 || cursors[1] != "qwen3.8-max" {
		t.Errorf("requested cursors %q, want the first page then after qwen3.8-max", cursors)
	}
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	first := models[0]
	if first.ID() != "qwen3.8-max" || first.DisplayName() != "Qwen 3.8 Max" ||
		first.ContextWindow() != 1000000 || first.MaxOutputTokens() != 131072 || first.CreatedAt().Year() != 2026 {
		t.Errorf("first model = %q %q ctx=%d out=%d at %v", first.ID(), first.DisplayName(), first.ContextWindow(), first.MaxOutputTokens(), first.CreatedAt())
	}
	if !models[1].CreatedAt().IsZero() {
		t.Errorf("an epoch release date means unknown, got %v", models[1].CreatedAt())
	}
}

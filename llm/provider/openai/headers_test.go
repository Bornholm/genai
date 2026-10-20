package openai_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	providerenv "github.com/bornholm/genai/llm/provider/env"
	"github.com/bornholm/genai/llm/provider/openai"
)

func TestChatCompletion_SendsTheConfiguredHeadersAndTheSessionID(t *testing.T) {
	var got http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer server.Close()

	client, err := provider.Create(context.Background(), provider.WithChatCompletion(openai.Name, openai.Options{
		CommonOptions: provider.CommonOptions{
			BaseURL:       server.URL,
			APIKey:        "sk-test",
			Model:         "glm-5.3",
			Headers:       map[string]string{"User-Agent": "xolo/1.0"},
			SessionHeader: "x-opencode-session",
		},
	}))
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	_, err = client.ChatCompletion(context.Background(),
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithSessionID("session-1"),
	)
	if err != nil {
		t.Fatalf("chat completion: %v", err)
	}

	if got.Get("User-Agent") != "xolo/1.0" {
		t.Errorf("user agent = %q", got.Get("User-Agent"))
	}
	if got.Get("X-Opencode-Session") != "session-1" {
		t.Errorf("session header = %q", got.Get("X-Opencode-Session"))
	}
	if got.Get("Authorization") != "Bearer sk-test" {
		t.Errorf("the SDK's own headers must be kept, got %v", got)
	}
}

func TestOptions_ReadTheHeadersFromTheEnvironment(t *testing.T) {
	env := map[string]string{
		"TEST_CHAT_COMPLETION_PROVIDER":              "openai",
		"TEST_CHAT_COMPLETION_OPENAI_HEADERS":        "User-Agent:xolo/1.0,X-Team:core",
		"TEST_CHAT_COMPLETION_OPENAI_SESSION_HEADER": "x-opencode-session",
	}
	for name, value := range env {
		os.Setenv(name, value)
		defer os.Unsetenv(name)
	}

	opts, err := provider.NewOptions(providerenv.With("TEST_"))
	if err != nil {
		t.Fatal(err)
	}
	typed := opts.ChatCompletion.Specific.(*openai.Options)
	if typed.Headers["User-Agent"] != "xolo/1.0" || typed.Headers["X-Team"] != "core" || len(typed.Headers) != 2 {
		t.Errorf("headers = %v", typed.Headers)
	}
	if typed.SessionHeader != "x-opencode-session" {
		t.Errorf("session header = %q", typed.SessionHeader)
	}
}

package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/mistral"
	"github.com/bornholm/genai/llm/provider/openai"
)

// newOpenAIBackend serves the proxy from a client of the named OpenAI
// compatible provider whose upstream records the request body it receives and
// answers with a short completion.
func newOpenAIBackend(t *testing.T, name provider.Name) (*Server, func() map[string]any) {
	t.Helper()

	var (
		mu   sync.Mutex
		body map[string]any
	)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		mu.Lock()
		_ = json.Unmarshal(raw, &body)
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(upstream.Close)

	common := provider.CommonOptions{BaseURL: upstream.URL, APIKey: "test", Model: "m"}
	var opt provider.OptionFunc
	switch name {
	case openai.Name:
		opt = provider.WithChatCompletion(name, openai.Options{CommonOptions: common})
	case mistral.Name:
		opt = provider.WithChatCompletion(name, mistral.Options{CommonOptions: common})
	default:
		t.Fatalf("unknown provider %s", name)
	}
	client, err := provider.Create(context.Background(), opt)
	if err != nil {
		t.Fatalf("could not create the %s client: %+v", name, err)
	}

	return NewServer(WithHook(&resolverHook{client: client, model: "m"})), func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return body
	}
}

func TestOpenAIBackends_KeepANamedToolChoiceOnMessages(t *testing.T) {
	for _, name := range []provider.Name{openai.Name, mistral.Name} {
		t.Run(string(name), func(t *testing.T) {
			server, upstreamBody := newOpenAIBackend(t, name)

			reqBody := `{"model":"m","max_tokens":100,
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"name":"weather","input_schema":{"type":"object"}}],
				"tool_choice":{"type":"tool","name":"weather","disable_parallel_tool_use":true}}`
			w := httptest.NewRecorder()
			server.handleMessages(w, buildMessagesRequest(t, "/v1/messages", reqBody))

			if w.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
			}

			body := upstreamBody()
			want := map[string]any{"type": "function", "function": map[string]any{"name": "weather"}}
			if got := body["tool_choice"]; !reflect.DeepEqual(got, want) {
				t.Errorf("tool_choice = %v, want %v", got, want)
			}
			if got := body["parallel_tool_calls"]; got != false {
				t.Errorf("parallel_tool_calls = %v, want false", got)
			}
		})
	}
}

func TestOpenAIBackend_KeepsAToolChoiceNamingAnUndeclaredToolRequired(t *testing.T) {
	server, upstreamBody := newOpenAIBackend(t, openai.Name)

	reqBody := `{"model":"m","max_tokens":100,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"weather","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"tool","name":"forecast"}}`
	w := httptest.NewRecorder()
	server.handleMessages(w, buildMessagesRequest(t, "/v1/messages", reqBody))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
	}
	if got := upstreamBody()["tool_choice"]; got != "required" {
		t.Errorf("tool_choice = %v, want required", got)
	}
}

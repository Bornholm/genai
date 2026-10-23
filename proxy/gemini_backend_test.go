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
	"github.com/bornholm/genai/llm/provider/gemini"
)

// newGeminiBackend serves the proxy from a gemini client whose upstream
// records the request body it receives and answers with a short completion.
func newGeminiBackend(t *testing.T) (*Server, func() map[string]any) {
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
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`)
	}))
	t.Cleanup(upstream.Close)

	client, err := provider.Create(context.Background(), provider.WithChatCompletion(gemini.Name, gemini.Options{
		CommonOptions: provider.CommonOptions{BaseURL: upstream.URL + "/v1beta", APIKey: "test", Model: "models/gemini-3-pro"},
	}))
	if err != nil {
		t.Fatalf("could not create the gemini client: %+v", err)
	}

	return NewServer(WithHook(&resolverHook{client: client, model: "models/gemini-3-pro"})), func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return body
	}
}

func TestGeminiBackend_TranslatesOpenAIFields(t *testing.T) {
	server, upstreamBody := newGeminiBackend(t)

	reqBody := `{"model":"gemini-3-pro","max_tokens":100,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}],
		"tool_choice":{"type":"function","function":{"name":"weather"}},
		"parallel_tool_calls":false,
		"stop":["END"],
		"top_p":0.9,
		"user":"u1",
		"frequency_penalty":0,
		"logprobs":true}`
	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildMessagesRequest(t, "/v1/chat/completions", reqBody))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
	}

	body := upstreamBody()
	for _, key := range []string{"stop", "top_p", "user", "tool_choice", "parallel_tool_calls", "frequency_penalty", "logprobs"} {
		if _, ok := body[key]; ok {
			t.Errorf("%s reached generateContent: %v", key, body)
		}
	}
	config, _ := body["generationConfig"].(map[string]any)
	if !reflect.DeepEqual(config["stopSequences"], []any{"END"}) || config["topP"] != 0.9 {
		t.Errorf("generationConfig = %v, want stopSequences [END] and topP 0.9", config)
	}
	want := map[string]any{"functionCallingConfig": map[string]any{"mode": "ANY", "allowedFunctionNames": []any{"weather"}}}
	if got := body["toolConfig"]; !reflect.DeepEqual(got, want) {
		t.Errorf("toolConfig = %v, want %v", got, want)
	}
}

func TestGeminiBackend_ForcesANamedToolOnMessages(t *testing.T) {
	server, upstreamBody := newGeminiBackend(t)

	reqBody := `{"model":"gemini-3-pro","max_tokens":100,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"weather","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"tool","name":"weather","disable_parallel_tool_use":true},
		"stop_sequences":["END"],
		"top_p":0.9}`
	w := httptest.NewRecorder()
	server.handleMessages(w, buildMessagesRequest(t, "/v1/messages", reqBody))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
	}

	body := upstreamBody()
	for _, key := range []string{"stop", "top_p", "tool_choice", "parallel_tool_calls"} {
		if _, ok := body[key]; ok {
			t.Errorf("%s reached generateContent: %v", key, body)
		}
	}
	config, _ := body["generationConfig"].(map[string]any)
	if !reflect.DeepEqual(config["stopSequences"], []any{"END"}) || config["topP"] != 0.9 {
		t.Errorf("generationConfig = %v, want stopSequences [END] and topP 0.9", config)
	}
	want := map[string]any{"functionCallingConfig": map[string]any{"mode": "ANY", "allowedFunctionNames": []any{"weather"}}}
	if got := body["toolConfig"]; !reflect.DeepEqual(got, want) {
		t.Errorf("toolConfig = %v, want %v", got, want)
	}
}

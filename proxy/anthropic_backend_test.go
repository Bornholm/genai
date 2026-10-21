package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/anthropic"
)

// newAnthropicBackend serves the proxy from an anthropic client whose upstream
// records the request body it receives and answers with a short message.
func newAnthropicBackend(t *testing.T) (*Server, func() map[string]any) {
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

		// The provider streams even a blocking completion.
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_start\n"+
			`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}}`+"\n\n"+
			"event: content_block_start\n"+
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n"+
			"event: content_block_delta\n"+
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`+"\n\n"+
			"event: content_block_stop\n"+
			`data: {"type":"content_block_stop","index":0}`+"\n\n"+
			"event: message_delta\n"+
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`+"\n\n"+
			"event: message_stop\n"+
			`data: {"type":"message_stop"}`+"\n\n")
	}))
	t.Cleanup(upstream.Close)

	// The anthropic client only does chat; the nil interfaces complete llm.Client.
	client := struct {
		*anthropic.ChatCompletionClient
		llm.EmbeddingsClient
		llm.TranscriptionClient
	}{ChatCompletionClient: anthropic.NewChatCompletionClient(anthropicsdk.NewClient(
		option.WithBaseURL(upstream.URL),
		option.WithAPIKey("test"),
		option.WithMaxRetries(0),
	), "claude-sonnet-5", 100)}

	return NewServer(WithHook(&resolverHook{client: client, model: "claude-sonnet-5"})), func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return body
	}
}

func TestAnthropicBackend_TranslatesOpenAIFields(t *testing.T) {
	server, upstreamBody := newAnthropicBackend(t)

	reqBody := `{"model":"claude-sonnet-5","max_tokens":100,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}],
		"tool_choice":{"type":"function","function":{"name":"weather"}},
		"parallel_tool_calls":false,
		"stop":["END"],
		"user":"u1",
		"frequency_penalty":0.5,
		"logprobs":true}`
	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildMessagesRequest(t, "/v1/chat/completions", reqBody))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
	}

	body := upstreamBody()
	for _, key := range []string{"stop", "user", "parallel_tool_calls", "frequency_penalty", "logprobs"} {
		if _, ok := body[key]; ok {
			t.Errorf("%s reached the Messages API: %v", key, body)
		}
	}
	if got := body["stop_sequences"]; !reflect.DeepEqual(got, []any{"END"}) {
		t.Errorf("stop_sequences = %v, want [END]", got)
	}
	if got := body["metadata"]; !reflect.DeepEqual(got, map[string]any{"user_id": "u1"}) {
		t.Errorf("metadata = %v, want the user as user_id", got)
	}
	want := map[string]any{"type": "tool", "name": "weather", "disable_parallel_tool_use": true}
	if got := body["tool_choice"]; !reflect.DeepEqual(got, want) {
		t.Errorf("tool_choice = %v, want %v", got, want)
	}
}

func TestAnthropicBackend_KeepsStopSequencesOnMessages(t *testing.T) {
	server, upstreamBody := newAnthropicBackend(t)

	reqBody := `{"model":"claude-sonnet-5","max_tokens":100,
		"messages":[{"role":"user","content":"hi"}],
		"stop_sequences":["END"],
		"top_p":0.9}`
	w := httptest.NewRecorder()
	server.handleMessages(w, buildMessagesRequest(t, "/v1/messages", reqBody))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
	}

	body := upstreamBody()
	if _, ok := body["stop"]; ok {
		t.Errorf("stop reached the Messages API: %v", body)
	}
	if got := body["stop_sequences"]; !reflect.DeepEqual(got, []any{"END"}) {
		t.Errorf("stop_sequences = %v, want [END]", got)
	}
	if got := body["top_p"]; got != 0.9 {
		t.Errorf("top_p = %v, want 0.9", got)
	}
}

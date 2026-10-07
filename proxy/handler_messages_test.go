package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/anthropic"
)

// mockStreamingChatClient implements llm.ChatCompletionStreamingClient for testing.
type mockStreamingChatClient struct {
	chunks []llm.StreamChunk
	err    error
}

func (m *mockStreamingChatClient) ChatCompletion(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return nil, nil
}

func (m *mockStreamingChatClient) ChatCompletionStream(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	if m.err != nil {
		return nil, m.err
	}

	ch := make(chan llm.StreamChunk, len(m.chunks))
	for _, c := range m.chunks {
		ch <- c
	}
	close(ch)
	return ch, nil
}

func (m *mockStreamingChatClient) Embeddings(_ context.Context, _ []string, _ ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

func (m *mockStreamingChatClient) Transcription(_ context.Context, _ []byte, _ ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	return nil, nil
}

func buildMessagesRequest(t *testing.T, path, body string) *http.Request {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func TestHandleMessages_Success(t *testing.T) {
	msg := llm.NewMessage(llm.RoleAssistant, "Hello!")
	usage := llm.NewChatCompletionUsage(5, 3, 8)
	mockRes := llm.NewChatCompletionResponse(msg, usage)

	client := &mockChatClient{response: mockRes}
	resolver := &resolverHook{client: client, model: "claude-3-5-sonnet-20241022"}

	server := NewServer(WithHook(resolver))

	reqBody := `{"model":"claude-3-5-sonnet-20241022","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	server.handleMessages(w, buildMessagesRequest(t, "/v1/messages", reqBody))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("could not decode response: %v", err)
	}
	if resp["type"] != "message" {
		t.Errorf("type = %v, want message", resp["type"])
	}
	content := resp["content"].([]any)
	block := content[0].(map[string]any)
	if block["text"] != "Hello!" {
		t.Errorf("content[0].text = %v, want Hello!", block["text"])
	}
}

func TestHandleMessages_InvalidJSON(t *testing.T) {
	server := NewServer()
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString("not json"))
	server.handleMessages(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", w.Code, http.StatusBadRequest)
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("could not decode response: %v", err)
	}
	if resp["type"] != "error" {
		t.Errorf("type = %v, want error", resp["type"])
	}
	errObj, ok := resp["error"].(map[string]any)
	if !ok {
		t.Fatalf("error field should be an object, got %T", resp["error"])
	}
	if errObj["type"] != "invalid_request_error" {
		t.Errorf("error.type = %v, want invalid_request_error", errObj["type"])
	}
}

func TestHandleMessages_ModelNotFound(t *testing.T) {
	server := NewServer() // no resolvers, no default client
	w := httptest.NewRecorder()
	reqBody := `{"model":"unknown","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	server.handleMessages(w, buildMessagesRequest(t, "/v1/messages", reqBody))

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want %d", w.Code, http.StatusNotFound)
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("could not decode response: %v", err)
	}
	if resp["type"] != "error" {
		t.Errorf("type = %v, want error", resp["type"])
	}
}

func TestHandleMessages_Stream(t *testing.T) {
	chunks := []llm.StreamChunk{
		llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "Hello")),
		llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(10, 5, 15)),
	}
	client := &mockStreamingChatClient{chunks: chunks}
	resolver := &resolverHook{client: client, model: "claude-3-5-sonnet-20241022"}

	server := NewServer(WithHook(resolver))

	reqBody := `{"model":"claude-3-5-sonnet-20241022","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	server.handleMessages(w, buildMessagesRequest(t, "/v1/messages", reqBody))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
	}
	if ct := w.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type = %q, want text/event-stream", ct)
	}

	body := w.Body.String()
	for _, want := range []string{"event: message_start", "event: content_block_start", "event: content_block_delta", "event: message_stop"} {
		if !bytes.Contains([]byte(body), []byte(want)) {
			t.Errorf("response body missing %q:\n%s", want, body)
		}
	}
}

func TestHandleCountTokens(t *testing.T) {
	server := NewServer()

	reqBody := `{
		"model": "claude-3-5-sonnet-20241022",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hello, how are you?"}]
	}`
	w := httptest.NewRecorder()
	server.handleCountTokens(w, buildMessagesRequest(t, "/v1/messages/count_tokens", reqBody))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("could not decode response: %v", err)
	}
	tokens, ok := resp["input_tokens"].(float64)
	if !ok || tokens <= 0 {
		t.Errorf("input_tokens = %v, want > 0", resp["input_tokens"])
	}
}

func TestHandleCountTokens_InvalidJSON(t *testing.T) {
	server := NewServer()
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewBufferString("not json"))
	server.handleCountTokens(w, req)

	// count_tokens uses a heuristic on the raw body and tolerates invalid
	// JSON by falling back to a byte-length estimate.
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
	}
}

func TestHandleMessages_InvalidCacheControlIsBadRequest(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("an invalid request reached the upstream")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()

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
	server := NewServer(WithHook(&resolverHook{client: client, model: "claude-sonnet-5"}))

	// Five breakpoints, one more than the API accepts.
	reqBody := `{"model":"claude-sonnet-5","max_tokens":100,
		"system":[
			{"type":"text","text":"one","cache_control":{"type":"ephemeral"}},
			{"type":"text","text":"two","cache_control":{"type":"ephemeral"}}
		],
		"messages":[
			{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]},
			{"role":"assistant","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]},
			{"role":"user","content":[{"type":"text","text":"again","cache_control":{"type":"ephemeral"}}]}
		]}`
	w := httptest.NewRecorder()
	server.handleMessages(w, buildMessagesRequest(t, "/v1/messages", reqBody))

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusBadRequest, w.Body.String())
	}
	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("could not decode response: %v", err)
	}
	if errObj, _ := resp["error"].(map[string]any); errObj["type"] != "invalid_request_error" {
		t.Errorf("error = %v, want type invalid_request_error", resp["error"])
	}
}

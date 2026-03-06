package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bornholm/genai/llm"
)

// mockChatClient implements llm.ChatCompletionClient for testing.
type mockChatClient struct {
	response llm.ChatCompletionResponse
	err      error
}

func (m *mockChatClient) ChatCompletion(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return m.response, m.err
}

func (m *mockChatClient) ChatCompletionStream(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	return nil, nil
}

func (m *mockChatClient) Embeddings(_ context.Context, _ []string, _ ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

// resolverHook provides a fixed client for the hook chain.
type resolverHook struct {
	client llm.Client
	model  string
}

func (r *resolverHook) Name() string     { return "test.resolver" }
func (r *resolverHook) Priority() int    { return 0 }
func (r *resolverHook) ResolveModel(_ context.Context, _ *ProxyRequest) (llm.Client, string, error) {
	return r.client, r.model, nil
}

func buildChatRequest(t *testing.T, body string) *http.Request {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func TestHandleChatCompletions_Success(t *testing.T) {
	msg := llm.NewMessage(llm.RoleAssistant, "Hello!")
	usage := llm.NewChatCompletionUsage(5, 3, 8)
	mockRes := llm.NewChatCompletionResponse(msg, usage)

	client := &mockChatClient{response: mockRes}
	resolver := &resolverHook{client: client, model: "gpt-4"}

	server := NewServer(WithHook(resolver))

	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", w.Code, http.StatusOK, w.Body.String())
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("could not decode response: %v", err)
	}
	if resp["object"] != "chat.completion" {
		t.Errorf("object = %v", resp["object"])
	}
}

func TestHandleChatCompletions_InvalidJSON(t *testing.T) {
	server := NewServer()
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString("not json"))
	server.handleChatCompletions(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", w.Code, http.StatusBadRequest)
	}
}

func TestHandleChatCompletions_ModelNotFound(t *testing.T) {
	server := NewServer() // no resolvers, no default client
	w := httptest.NewRecorder()
	reqBody := `{"model":"unknown","messages":[{"role":"user","content":"hi"}]}`
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want %d", w.Code, http.StatusNotFound)
	}
}

func TestHandleChatCompletions_PreRequestShortCircuit(t *testing.T) {
	blockerResp := &ProxyResponse{
		StatusCode: http.StatusForbidden,
		Body:       ErrorResponse{Error: *NewBadRequestError("blocked")},
	}
	blocker := &stubPreHook{
		hookName:     "blocker",
		hookPriority: 1,
		result:       &HookResult{Response: blockerResp},
	}

	server := NewServer(WithHook(blocker))
	w := httptest.NewRecorder()
	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"bad request"}]}`
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	if w.Code != http.StatusForbidden {
		t.Errorf("status = %d, want %d", w.Code, http.StatusForbidden)
	}
}

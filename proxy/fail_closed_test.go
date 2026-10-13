package proxy

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

// recordingClient answers like mockChatClient and records whether it was
// called, to tell a request that reached the backend from one that did not.
type recordingClient struct {
	mockChatClient
	called bool
}

func (c *recordingClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	c.called = true
	return c.mockChatClient.ChatCompletion(ctx, funcs...)
}

func newRecordingClient() *recordingClient {
	return &recordingClient{mockChatClient: mockChatClient{
		response: llm.NewChatCompletionResponse(llm.NewMessage(llm.RoleAssistant, "Hello!"), llm.NewChatCompletionUsage(5, 3, 8)),
	}}
}

const chatBody = `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`

func TestHandleChatCompletions_PreRequestHookErrorBlocksTheRequest(t *testing.T) {
	client := newRecordingClient()
	hook := &stubPreHook{hookName: "quota", err: errors.New("quota store at db.internal:5432 unreachable")}
	server := NewServer(WithHook(hook), WithDefaultClient(client))

	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildChatRequest(t, chatBody))

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want %d; body: %s", w.Code, http.StatusInternalServerError, w.Body.String())
	}
	if client.called {
		t.Error("the backend was called although a pre-request hook failed")
	}
	if strings.Contains(w.Body.String(), "db.internal") {
		t.Errorf("body = %s, want the hook error kept out of the response", w.Body.String())
	}
}

func TestHandleChatCompletions_PreRequestHookRefusesWithItsOwnStatus(t *testing.T) {
	client := newRecordingClient()
	hook := &stubPreHook{hookName: "acl", err: &APIError{StatusCode: http.StatusForbidden, Type: "permission_error", Message: "not allowed"}}
	server := NewServer(WithHook(hook), WithDefaultClient(client))

	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildChatRequest(t, chatBody))

	if w.Code != http.StatusForbidden {
		t.Errorf("status = %d, want %d; body: %s", w.Code, http.StatusForbidden, w.Body.String())
	}
	if client.called {
		t.Error("the backend was called although a pre-request hook refused the request")
	}
}

func TestHandleMessages_PreRequestHookErrorBlocksTheRequest(t *testing.T) {
	client := newRecordingClient()
	hook := &stubPreHook{hookName: "quota", err: errors.New("quota store unreachable")}
	server := NewServer(WithHook(hook), WithDefaultClient(client))

	w := httptest.NewRecorder()
	server.handleMessages(w, buildMessagesRequest(t, "/v1/messages",
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want %d; body: %s", w.Code, http.StatusInternalServerError, w.Body.String())
	}
	if client.called {
		t.Error("the backend was called although a pre-request hook failed")
	}
}

func TestHandleChatCompletions_ResolverErrorDoesNotFallBackToTheDefaultClient(t *testing.T) {
	for name, err := range map[string]error{
		"plain error": errors.New("user u1 may not use gpt-4"),
		"api error":   &APIError{StatusCode: http.StatusForbidden, Type: "permission_error", Message: "not allowed"},
	} {
		t.Run(name, func(t *testing.T) {
			client := newRecordingClient()
			resolver := &stubResolver{hookName: "acl", err: err}
			server := NewServer(WithHook(resolver), WithDefaultClient(client))

			w := httptest.NewRecorder()
			server.handleChatCompletions(w, buildChatRequest(t, chatBody))

			if w.Code == http.StatusOK || client.called {
				t.Fatalf("status = %d, default client called = %v; want the resolver error to stop the request", w.Code, client.called)
			}
			var apiErr *APIError
			if errors.As(err, &apiErr) && w.Code != apiErr.StatusCode {
				t.Errorf("status = %d, want the resolver's %d", w.Code, apiErr.StatusCode)
			}
		})
	}
}

func TestHandleChatCompletions_UnknownModelFallsBackToTheDefaultClient(t *testing.T) {
	client := newRecordingClient()
	resolver := &stubResolver{hookName: "static", err: ErrModelNotFound}
	server := NewServer(WithHook(resolver), WithDefaultClient(client))

	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildChatRequest(t, chatBody))

	if w.Code != http.StatusOK || !client.called {
		t.Errorf("status = %d, default client called = %v; want the default client to answer", w.Code, client.called)
	}
}

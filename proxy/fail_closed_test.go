package proxy

import (
	"bytes"
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

func (c *recordingClient) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	c.called = true
	return c.mockChatClient.Embeddings(ctx, inputs, funcs...)
}

func newRecordingClient() *recordingClient {
	return &recordingClient{mockChatClient: mockChatClient{
		response: llm.NewChatCompletionResponse(llm.NewMessage(llm.RoleAssistant, "Hello!"), llm.NewChatCompletionUsage(5, 3, 8)),
	}}
}

// endpoint is one of the handlers a hook or a resolver error must stop.
type endpoint struct {
	name   string
	handle func(s *Server, w http.ResponseWriter, r *http.Request)
	path   string
	body   string
}

var endpoints = []endpoint{
	{"chat completions", (*Server).handleChatCompletions, "/v1/chat/completions",
		`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`},
	{"messages", (*Server).handleMessages, "/v1/messages",
		`{"model":"gpt-4","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`},
	{"embeddings", (*Server).handleEmbeddings, "/v1/embeddings",
		`{"model":"gpt-4","input":"hi"}`},
}

func (e endpoint) serve(s *Server) *httptest.ResponseRecorder {
	r := httptest.NewRequest(http.MethodPost, e.path, bytes.NewBufferString(e.body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	e.handle(s, w, r)
	return w
}

// answeringErrorHook answers every error with a fixed response.
type answeringErrorHook struct{ status int }

func (h *answeringErrorHook) Name() string  { return "test.error" }
func (h *answeringErrorHook) Priority() int { return 0 }
func (h *answeringErrorHook) OnError(_ context.Context, _ *ProxyRequest, _ error) (*HookResult, error) {
	return &HookResult{Response: &ProxyResponse{
		StatusCode: h.status,
		Body:       ErrorResponse{Error: APIError{Type: "test", Message: "answered by the error hook"}},
	}}, nil
}

func forbidden() *APIError {
	return &APIError{StatusCode: http.StatusForbidden, Type: "permission_error", Message: "not allowed"}
}

func TestPreRequestHookError_StopsTheRequest(t *testing.T) {
	for _, e := range endpoints {
		t.Run(e.name, func(t *testing.T) {
			client := newRecordingClient()
			hook := &stubPreHook{hookName: "quota", err: errors.New("quota store at db.internal:5432 unreachable")}

			w := e.serve(NewServer(WithHook(hook), WithDefaultClient(client)))

			if w.Code != http.StatusInternalServerError {
				t.Errorf("status = %d, want %d; body: %s", w.Code, http.StatusInternalServerError, w.Body.String())
			}
			if client.called {
				t.Error("the backend was called although a pre-request hook failed")
			}
			if strings.Contains(w.Body.String(), "db.internal") {
				t.Errorf("body = %s, want the hook error kept out of the response", w.Body.String())
			}
		})
	}
}

func TestPreRequestHookError_RefusesWithItsOwnStatus(t *testing.T) {
	for _, e := range endpoints {
		t.Run(e.name, func(t *testing.T) {
			client := newRecordingClient()
			hook := &stubPreHook{hookName: "acl", err: forbidden()}

			w := e.serve(NewServer(WithHook(hook), WithDefaultClient(client)))

			if w.Code != http.StatusForbidden || client.called {
				t.Errorf("status = %d, backend called = %v; want 403 without a backend call", w.Code, client.called)
			}
		})
	}
}

func TestPreRequestHookError_AnsweredByAnErrorHook(t *testing.T) {
	client := newRecordingClient()
	hook := &stubPreHook{hookName: "quota", err: errors.New("quota store unreachable")}
	server := NewServer(WithHook(hook), WithHook(&answeringErrorHook{status: http.StatusServiceUnavailable}), WithDefaultClient(client))

	w := endpoints[0].serve(server)

	if w.Code != http.StatusServiceUnavailable || client.called {
		t.Errorf("status = %d, backend called = %v; want the error hook's 503 without a backend call", w.Code, client.called)
	}
}

func TestResolverError_DoesNotFallBackToTheDefaultClient(t *testing.T) {
	for _, e := range endpoints {
		t.Run(e.name, func(t *testing.T) {
			for name, tc := range map[string]struct {
				err  error
				want int
			}{
				"plain error": {errors.New("user u1 may not use gpt-4 (acl at db.internal)"), http.StatusInternalServerError},
				"api error":   {forbidden(), http.StatusForbidden},
			} {
				t.Run(name, func(t *testing.T) {
					client := newRecordingClient()
					resolver := &stubResolver{hookName: "acl", err: tc.err}

					w := e.serve(NewServer(WithHook(resolver), WithDefaultClient(client)))

					if w.Code != tc.want || client.called {
						t.Errorf("status = %d, default client called = %v; want %d without a backend call", w.Code, client.called, tc.want)
					}
					if strings.Contains(w.Body.String(), "db.internal") {
						t.Errorf("body = %s, want the resolver error kept out of the response", w.Body.String())
					}
				})
			}
		})
	}
}

func TestResolverError_AnsweredByAnErrorHook(t *testing.T) {
	client := newRecordingClient()
	resolver := &stubResolver{hookName: "acl", err: errors.New("acl store unreachable")}
	server := NewServer(WithHook(resolver), WithHook(&answeringErrorHook{status: http.StatusServiceUnavailable}), WithDefaultClient(client))

	w := endpoints[0].serve(server)

	if w.Code != http.StatusServiceUnavailable || client.called {
		t.Errorf("status = %d, backend called = %v; want the error hook's 503 without a backend call", w.Code, client.called)
	}
}

func TestUnknownModel_FallsBackToTheDefaultClient(t *testing.T) {
	client := newRecordingClient()
	resolver := &stubResolver{hookName: "static", err: ErrModelNotFound}

	w := endpoints[0].serve(NewServer(WithHook(resolver), WithDefaultClient(client)))

	if w.Code != http.StatusOK || !client.called {
		t.Errorf("status = %d, default client called = %v; want the default client to answer", w.Code, client.called)
	}
}

func TestAuthExtractorError_KeepsItsCauseOutOfTheResponse(t *testing.T) {
	for _, e := range endpoints {
		t.Run(e.name, func(t *testing.T) {
			client := newRecordingClient()
			extractor := func(*http.Request) (string, error) {
				return "", errors.New("could not fetch keys from https://keys.internal/jwks")
			}

			w := e.serve(NewServer(WithAuthExtractor(extractor), WithDefaultClient(client)))

			if w.Code != http.StatusUnauthorized || client.called {
				t.Errorf("status = %d, backend called = %v; want 401 without a backend call", w.Code, client.called)
			}
			if strings.Contains(w.Body.String(), "keys.internal") {
				t.Errorf("body = %s, want the extractor error kept out of the response", w.Body.String())
			}
		})
	}
}

func TestAuthExtractorError_RefusesWithItsOwnAPIError(t *testing.T) {
	extractor := func(*http.Request) (string, error) {
		return "", NewUnauthorizedError("missing API key")
	}

	w := endpoints[0].serve(NewServer(WithAuthExtractor(extractor), WithDefaultClient(newRecordingClient())))

	if w.Code != http.StatusUnauthorized || !strings.Contains(w.Body.String(), "missing API key") {
		t.Errorf("status = %d, body = %s; want the extractor's own 401", w.Code, w.Body.String())
	}
}

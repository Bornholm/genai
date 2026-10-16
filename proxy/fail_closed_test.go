package proxy

import (
	"bytes"
	"context"
	"encoding/json"
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

func TestAPIErrorWithoutStatus_AnswersWith500(t *testing.T) {
	for _, e := range endpoints {
		t.Run(e.name, func(t *testing.T) {
			hook := &stubPreHook{hookName: "acl", err: &APIError{Type: "permission_error", Message: "not allowed"}}

			w := e.serve(NewServer(WithHook(hook), WithDefaultClient(newRecordingClient())))

			if w.Code != http.StatusInternalServerError {
				t.Errorf("status = %d, want %d; body: %s", w.Code, http.StatusInternalServerError, w.Body.String())
			}
		})
	}
}

func TestHookResponseWithoutStatus_AnswersWith500(t *testing.T) {
	hook := &stubPreHook{hookName: "quota", result: &HookResult{Response: &ProxyResponse{
		Body: ErrorResponse{Error: APIError{Type: "rate_limit_error", Message: "quota exceeded"}},
	}}}

	w := endpoints[0].serve(NewServer(WithHook(hook), WithDefaultClient(newRecordingClient())))

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want %d", w.Code, http.StatusInternalServerError)
	}
}

// assertAnthropicError checks that body is an Anthropic error envelope.
func assertAnthropicError(t *testing.T, body []byte, wantType, wantMessage string) {
	t.Helper()
	var got AnthropicErrorResponse
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("body = %s, not JSON: %v", body, err)
	}
	if got.Type != "error" || got.Error.Type != wantType || got.Error.Message != wantMessage {
		t.Errorf("body = %s, want an Anthropic %q error saying %q", body, wantType, wantMessage)
	}
}

func TestMessages_HookResponsesUseTheAnthropicEnvelope(t *testing.T) {
	messages := endpoints[1]
	quotaExceeded := &HookResult{Response: &ProxyResponse{
		StatusCode: http.StatusTooManyRequests,
		Body:       ErrorResponse{Error: APIError{Type: "rate_limit_error", Message: "quota exceeded"}},
	}}

	t.Run("pre-request refusal", func(t *testing.T) {
		hook := &stubPreHook{hookName: "quota", result: quotaExceeded}

		w := messages.serve(NewServer(WithHook(hook), WithDefaultClient(newRecordingClient())))

		if w.Code != http.StatusTooManyRequests {
			t.Errorf("status = %d, want %d", w.Code, http.StatusTooManyRequests)
		}
		assertAnthropicError(t, w.Body.Bytes(), "rate_limit_error", "quota exceeded")
	})

	t.Run("error hook after a resolver error", func(t *testing.T) {
		resolver := &stubResolver{hookName: "acl", err: errors.New("acl store unreachable")}
		server := NewServer(WithHook(resolver), WithHook(&answeringErrorHook{status: http.StatusServiceUnavailable}), WithDefaultClient(newRecordingClient()))

		w := messages.serve(server)

		if w.Code != http.StatusServiceUnavailable {
			t.Errorf("status = %d, want %d", w.Code, http.StatusServiceUnavailable)
		}
		assertAnthropicError(t, w.Body.Bytes(), "api_error", "answered by the error hook")
	})

	t.Run("error hook after a stream that failed to open", func(t *testing.T) {
		client := &erroringStreamClient{err: errors.New("upstream unreachable")}
		server := NewServer(WithHook(&answeringErrorHook{status: http.StatusServiceUnavailable}), WithDefaultClient(client))
		r := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(
			`{"model":"gpt-4","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
		w := httptest.NewRecorder()

		server.handleMessages(w, r)

		if w.Code != http.StatusServiceUnavailable {
			t.Errorf("status = %d, want %d; body: %s", w.Code, http.StatusServiceUnavailable, w.Body.String())
		}
		assertAnthropicError(t, w.Body.Bytes(), "api_error", "answered by the error hook")
	})

	t.Run("other responses are left as is", func(t *testing.T) {
		hook := &stubPreHook{hookName: "cache", result: &HookResult{Response: &ProxyResponse{
			StatusCode: http.StatusOK,
			Body:       map[string]any{"cached": true},
		}}}

		w := messages.serve(NewServer(WithHook(hook), WithDefaultClient(newRecordingClient())))

		if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"cached":true`) {
			t.Errorf("status = %d, body = %s; want the hook's response unchanged", w.Code, w.Body.String())
		}
	})
}

func TestBackendError_KeepsItsCauseOutOfTheResponse(t *testing.T) {
	for _, e := range endpoints {
		t.Run(e.name, func(t *testing.T) {
			client := &mockChatClient{err: errors.New("dial tcp 10.0.3.7:443: connect: connection refused")}

			w := e.serve(NewServer(WithDefaultClient(client)))

			if w.Code != http.StatusInternalServerError {
				t.Errorf("status = %d, want %d", w.Code, http.StatusInternalServerError)
			}
			if strings.Contains(w.Body.String(), "10.0.3.7") {
				t.Errorf("body = %s, want the backend error kept out of the response", w.Body.String())
			}
		})
	}
}

func TestBackendError_KeepsAnAPIError(t *testing.T) {
	for _, e := range endpoints {
		t.Run(e.name, func(t *testing.T) {
			client := &mockChatClient{err: &APIError{StatusCode: http.StatusTooManyRequests, Type: "rate_limit_error", Message: "slow down"}}

			w := e.serve(NewServer(WithDefaultClient(client)))

			if w.Code != http.StatusTooManyRequests {
				t.Errorf("status = %d, want %d", w.Code, http.StatusTooManyRequests)
			}
			if !strings.Contains(w.Body.String(), "slow down") {
				t.Errorf("body = %s, want the message of the *APIError", w.Body.String())
			}
		})
	}
}

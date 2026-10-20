package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bornholm/genai/llm"
)

// sessionClient records the session ID a chat completion carried, as an
// option and on its context.
type sessionClient struct {
	mockChatClient
	gotOption, gotContext string
}

func (c *sessionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	c.gotOption = llm.NewChatCompletionOptions(funcs...).SessionID
	c.gotContext = llm.SessionIDFromContext(ctx)
	return llm.NewChatCompletionResponse(llm.NewMessage(llm.RoleAssistant, "ok"), llm.NewChatCompletionUsage(1, 1, 2)), nil
}

// sessionRelay records the session ID a relayed request carried on its
// context.
type sessionRelay struct {
	relayClient
	gotContext string
}

func (c *sessionRelay) RelayMessages(ctx context.Context, body []byte, header http.Header) (<-chan llm.StreamChunk, error) {
	c.gotContext = llm.SessionIDFromContext(ctx)
	return c.relayClient.RelayMessages(ctx, body, header)
}

// sessionHook replaces the session ID of every request.
type sessionHook struct{ id string }

func (h *sessionHook) Name() string  { return "test.session" }
func (h *sessionHook) Priority() int { return 0 }
func (h *sessionHook) PreRequest(_ context.Context, req *ProxyRequest) (*HookResult, error) {
	req.SessionID = h.id
	return nil, nil
}

func TestHandleChatCompletions_PassesTheClientSessionUpstream(t *testing.T) {
	for _, header := range []string{"X-Session-Id", "X-Claude-Code-Session-Id", "session_id", "x-opencode-session"} {
		t.Run(header, func(t *testing.T) {
			client := &sessionClient{}
			server := NewServer(WithHook(&resolverHook{client: client, model: "m"}))

			req := buildChatRequest(t, `{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
			req.Header.Set(header, "conversation-1")
			w := httptest.NewRecorder()
			server.handleChatCompletions(w, req)

			if w.Code != http.StatusOK {
				t.Fatalf("status = %d; body: %s", w.Code, w.Body.String())
			}
			if client.gotOption != "conversation-1" || client.gotContext != "conversation-1" {
				t.Errorf("session option = %q, context = %q, want conversation-1 for both", client.gotOption, client.gotContext)
			}
		})
	}
}

func TestHandleChatCompletions_SendsNoSessionWithoutOne(t *testing.T) {
	client := &sessionClient{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "m"}))

	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildChatRequest(t, `{"model":"m","messages":[{"role":"user","content":"hi"}]}`))

	if client.gotOption != "" || client.gotContext != "" {
		t.Errorf("session option = %q, context = %q, want none", client.gotOption, client.gotContext)
	}
}

func TestHandleMessages_PassesTheHookSessionUpstream(t *testing.T) {
	client := &sessionClient{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "m"}), WithHook(&sessionHook{id: "from-hook"}))

	req := buildMessagesRequest(t, "/v1/messages", `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`)
	req.Header.Set("X-Claude-Code-Session-Id", "from-client")
	w := httptest.NewRecorder()
	server.handleMessages(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d; body: %s", w.Code, w.Body.String())
	}
	if client.gotOption != "from-hook" || client.gotContext != "from-hook" {
		t.Errorf("session option = %q, context = %q, want from-hook for both", client.gotOption, client.gotContext)
	}
}

func TestHandleMessages_PassesTheSessionToARelay(t *testing.T) {
	client := &sessionRelay{relayClient: relayClient{events: [][]byte{
		[]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
	}}}
	server := NewServer(WithHook(&resolverHook{client: client, model: "m"}))

	req := buildMessagesRequest(t, "/messages", relayRequestBody)
	req.Header.Set("X-Claude-Code-Session-Id", "conversation-1")
	w := httptest.NewRecorder()
	server.handleMessages(w, req)

	if client.gotContext != "conversation-1" {
		t.Errorf("relay context session = %q, want conversation-1", client.gotContext)
	}
}

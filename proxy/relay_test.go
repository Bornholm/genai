package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

// relayClient is a streaming client that also relays Messages requests,
// recording what it was asked to relay.
type relayClient struct {
	mockStreamingChatClient
	events [][]byte
	err    error

	gotBody   []byte
	gotHeader http.Header
}

func (c *relayClient) RelayMessages(_ context.Context, body []byte, header http.Header) (<-chan llm.StreamChunk, error) {
	c.gotBody, c.gotHeader = body, header
	if c.err != nil {
		return nil, c.err
	}
	ch := make(chan llm.StreamChunk, len(c.events))
	for i, event := range c.events {
		ch <- llm.NewRawEventChunk(event, llm.NewChatCompletionUsage(10, int64(i), 10+int64(i)), i == len(c.events)-1)
	}
	close(ch)
	return ch, nil
}

type usageRecorder struct{ res *ProxyResponse }

func (h *usageRecorder) Name() string  { return "test.usage" }
func (h *usageRecorder) Priority() int { return 0 }
func (h *usageRecorder) PostResponse(_ context.Context, _ *ProxyRequest, res *ProxyResponse) (*HookResult, error) {
	h.res = res
	return nil, nil
}

const relayRequestBody = `{"model":"claude","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}],"safeguards":[{"type":"dangerous_tool_use"}]}`

func TestHandleMessages_RelaysStreamVerbatim(t *testing.T) {
	events := [][]byte{
		[]byte("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"),
		[]byte("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"safeguard_results\":[]}}\n\n"),
		[]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
	}
	client := &relayClient{events: events}
	usage := &usageRecorder{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "claude-real"}), WithHook(usage))

	req := buildMessagesRequest(t, "/messages", relayRequestBody)
	req.Header.Set("Anthropic-Beta", "dangerous-tool-use-2026-09-03")
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)

	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status %d content-type %q: %s", w.Code, w.Header().Get("Content-Type"), w.Body.String())
	}
	want := string(events[0]) + string(events[1]) + string(events[2])
	if w.Body.String() != want {
		t.Errorf("events not relayed verbatim:\n%s", w.Body.String())
	}
	if string(client.gotBody) != relayRequestBody {
		t.Errorf("relayed body = %s", client.gotBody)
	}
	if client.gotHeader.Get("Anthropic-Beta") != "dangerous-tool-use-2026-09-03" {
		t.Errorf("anthropic-beta not handed to the relay: %v", client.gotHeader)
	}
	if usage.res == nil || usage.res.Interruption != nil || usage.res.TokensUsed.CompletionTokens != 2 {
		t.Errorf("post-response hooks got %+v", usage.res)
	}
}

func TestHandleMessages_RelayRejectionForwardedAsIs(t *testing.T) {
	const body = `{"type":"error","error":{"type":"invalid_request_error","message":"Input tag 'advisor_20260301' found"}}`
	httpErr := llm.NewHTTPError(http.StatusBadRequest, body)
	httpErr.Header = http.Header{"Retry-After": {"3"}, "Content-Length": {"999"}, "Anthropic-Organization-Id": {"org-operator"}}
	server := NewServer(WithHook(&resolverHook{client: &relayClient{err: httpErr}, model: "claude-real"}))

	w := httptest.NewRecorder()
	server.ServeHTTP(w, buildMessagesRequest(t, "/messages", relayRequestBody))

	if w.Code != http.StatusBadRequest || w.Body.String() != body {
		t.Errorf("got %d %s", w.Code, w.Body.String())
	}
	if w.Header().Get("Retry-After") != "3" || w.Header().Get("Content-Length") == "999" || w.Header().Get("Anthropic-Organization-Id") != "" {
		t.Errorf("headers = %v", w.Header())
	}
}

func TestHandleMessages_TruncatedRelayWritesNoClosingEvents(t *testing.T) {
	start := []byte("event: message_start\ndata: {\"type\":\"message_start\"}\n\n")
	usage := &usageRecorder{}
	server := NewServer(WithHook(&resolverHook{client: &truncatingRelay{event: start}, model: "m"}), WithHook(usage))

	w := httptest.NewRecorder()
	server.ServeHTTP(w, buildMessagesRequest(t, "/messages", relayRequestBody))

	if w.Body.String() != string(start) {
		t.Errorf("a truncated relay must not be closed on the client's behalf:\n%s", w.Body.String())
	}
	if usage.res == nil || usage.res.Interruption == nil || usage.res.Interruption.Cause != StreamInterruptionTruncated {
		t.Errorf("interruption = %+v", usage.res)
	}
}

type truncatingRelay struct {
	mockStreamingChatClient
	event []byte
}

func (c *truncatingRelay) RelayMessages(context.Context, []byte, http.Header) (<-chan llm.StreamChunk, error) {
	ch := make(chan llm.StreamChunk, 1)
	ch <- llm.NewRawEventChunk(c.event, nil, false)
	close(ch)
	return ch, nil
}

// hidingDecorator wraps a relaying client without passing the relay on.
type hidingDecorator struct{ *relayClient }

func (hidingDecorator) SupportsMessagesRelay() bool { return false }

func TestHandleMessages_HiddenRelayFallsBackToTranslation(t *testing.T) {
	inner := &relayClient{mockStreamingChatClient: mockStreamingChatClient{chunks: []llm.StreamChunk{
		llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "translated")),
		llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(1, 1, 2)),
	}}}
	server := NewServer(WithHook(&resolverHook{client: hidingDecorator{inner}, model: "m"}))

	w := httptest.NewRecorder()
	server.ServeHTTP(w, buildMessagesRequest(t, "/messages", relayRequestBody))

	if inner.gotBody != nil || !strings.Contains(w.Body.String(), "translated") {
		t.Errorf("expected the translated path, got relay=%v body=%s", inner.gotBody != nil, w.Body.String())
	}
}

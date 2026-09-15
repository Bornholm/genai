package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/bornholm/genai/llm"
)

// sseEvent is one Messages API server-sent event.
type sseEvent struct {
	name string
	data map[string]any
}

// fakeMessagesAPI records the requests it receives and answers each with
// the scripted events, or with an HTTP error.
type fakeMessagesAPI struct {
	mu       sync.Mutex
	requests []capturedRequest
	events   []sseEvent
	status   int
	errBody  string
}

type capturedRequest struct {
	path string
	body map[string]any
}

func (f *fakeMessagesAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)

	f.mu.Lock()
	f.requests = append(f.requests, capturedRequest{path: r.URL.Path, body: body})
	f.mu.Unlock()

	if f.status != 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(f.status)
		_, _ = io.WriteString(w, f.errBody)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	for _, ev := range f.events {
		data, _ := json.Marshal(ev.data)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.name, data)
	}
}

func (f *fakeMessagesAPI) lastRequest(t *testing.T) capturedRequest {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) == 0 {
		t.Fatal("no request received")
	}
	return f.requests[len(f.requests)-1]
}

func newTestClient(t *testing.T, fake *fakeMessagesAPI, baseURLSuffix string) *ChatCompletionClient {
	t.Helper()
	server := httptest.NewServer(fake)
	t.Cleanup(server.Close)

	client := anthropicsdk.NewClient(
		option.WithBaseURL(normalizeBaseURL(server.URL+baseURLSuffix)),
		option.WithAPIKey("test-key"),
		option.WithMaxRetries(0),
	)
	return NewChatCompletionClient(client, "claude-sonnet-5", DefaultMaxTokens)
}

// scriptedReply is a full response: a thinking block, a text block, a
// tool_use block, with prompt caching counters in the usage.
func scriptedReply() []sseEvent {
	return []sseEvent{
		{"message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
				"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]any{
					"input_tokens": 10, "output_tokens": 1,
					"cache_read_input_tokens": 100, "cache_creation_input_tokens": 50,
				},
			},
		}},
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "thinking", "thinking": "", "signature": ""}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "thinking_delta", "thinking": "Let me "}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "thinking_delta", "thinking": "think."}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "signature_delta", "signature": "sig-1"}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}},
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 1,
			"content_block": map[string]any{"type": "text", "text": ""}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 1,
			"delta": map[string]any{"type": "text_delta", "text": "Hello, "}}},
		{"content_block_delta", map[string]any{"index": 1, "type": "content_block_delta",
			"delta": map[string]any{"type": "text_delta", "text": "world"}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 1}},
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 2,
			"content_block": map[string]any{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": map[string]any{}}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 2,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": `{"location":`}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 2,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": `"Paris"}`}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 2}},
		{"message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 20}}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}
}

func TestChatCompletion(t *testing.T) {
	fake := &fakeMessagesAPI{events: scriptedReply()}
	client := newTestClient(t, fake, "/v1")

	res, err := client.ChatCompletion(context.Background(),
		llm.WithMessages(
			llm.NewMessageWithCacheControl(llm.RoleSystem, "Be brief.", &llm.CacheControl{Type: "ephemeral"}),
			llm.NewMessage(llm.RoleUser, "Weather in Paris?"),
		),
	)
	if err != nil {
		t.Fatalf("ChatCompletion error: %+v", err)
	}

	req := fake.lastRequest(t)
	if req.path != "/v1/messages" {
		t.Errorf("a base URL ending in /v1 must still reach /v1/messages, got %s", req.path)
	}
	system, _ := req.body["system"].([]any)
	if len(system) != 1 {
		t.Fatalf("system prompt not hoisted: %v", req.body["system"])
	}
	if cc, _ := system[0].(map[string]any)["cache_control"].(map[string]any); cc["type"] != "ephemeral" {
		t.Errorf("cache_control did not reach the wire: %v", system[0])
	}
	if req.body["stream"] != true {
		t.Errorf("requests must be streamed, got stream=%v", req.body["stream"])
	}

	if res.Message().Content() != "Hello, world" {
		t.Errorf("unexpected content: %q", res.Message().Content())
	}
	if res.Message().Role() != llm.RoleAssistant {
		t.Errorf("unexpected role: %s", res.Message().Role())
	}

	rr, ok := res.(llm.ReasoningChatCompletionResponse)
	if !ok {
		t.Fatal("response does not implement ReasoningChatCompletionResponse")
	}
	if rr.Reasoning() != "Let me think." {
		t.Errorf("unexpected reasoning: %q", rr.Reasoning())
	}
	details := rr.ReasoningDetails()
	if len(details) != 1 || details[0].Signature != "sig-1" || details[0].Text != "Let me think." {
		t.Errorf("unexpected reasoning details: %+v", details)
	}
	if rm, ok := res.Message().(llm.ReasoningMessage); !ok || len(rm.ReasoningDetails()) != 1 {
		t.Error("the message must carry the reasoning details for replay")
	}

	toolCalls := res.ToolCalls()
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if toolCalls[0].ID() != "toolu_1" || toolCalls[0].Name() != "get_weather" {
		t.Errorf("unexpected tool call: %s %s", toolCalls[0].ID(), toolCalls[0].Name())
	}
	if params, _ := toolCalls[0].Parameters().(string); params != `{"location":"Paris"}` {
		t.Errorf("unexpected tool call parameters: %v", toolCalls[0].Parameters())
	}

	usage := res.Usage()
	if usage.PromptTokens() != 160 || usage.CompletionTokens() != 20 || usage.TotalTokens() != 180 {
		t.Errorf("unexpected usage: prompt=%d completion=%d total=%d", usage.PromptTokens(), usage.CompletionTokens(), usage.TotalTokens())
	}
	if cu, ok := usage.(interface{ CachedTokens() int64 }); !ok || cu.CachedTokens() != 100 {
		t.Errorf("cache reads not reported: %v", usage)
	}
	if cw, ok := usage.(llm.CacheCreationReportingUsage); !ok || cw.CacheCreationTokens() != 50 {
		t.Errorf("cache writes not reported: %v", usage)
	}
}

func TestChatCompletion_ExcludeReasoning(t *testing.T) {
	fake := &fakeMessagesAPI{events: scriptedReply()}
	client := newTestClient(t, fake, "")

	res, err := client.ChatCompletion(context.Background(),
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithReasoning(&llm.ReasoningOptions{Exclude: true}),
	)
	if err != nil {
		t.Fatalf("ChatCompletion error: %+v", err)
	}
	if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok && (rr.Reasoning() != "" || len(rr.ReasoningDetails()) != 0) {
		t.Errorf("reasoning must be withheld when excluded: %q", rr.Reasoning())
	}
	if res.Message().Content() != "Hello, world" {
		t.Errorf("content lost: %q", res.Message().Content())
	}
}

func TestChatCompletion_MapsHTTPErrors(t *testing.T) {
	fake := &fakeMessagesAPI{status: http.StatusTooManyRequests, errBody: `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`}
	client := newTestClient(t, fake, "")

	_, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(err, llm.ErrRateLimit) {
		t.Errorf("429 must be reported as ErrRateLimit, got %v", err)
	}
	var httpErr *llm.HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusTooManyRequests || !strings.Contains(httpErr.Body, "slow down") {
		t.Errorf("HTTPError not propagated: %v", err)
	}
	if !llm.IsRetryable(err) {
		t.Error("a 429 must be retryable")
	}
}

func TestChatCompletionStream(t *testing.T) {
	fake := &fakeMessagesAPI{events: scriptedReply()}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(),
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Weather in Paris?")),
	)
	if err != nil {
		t.Fatalf("ChatCompletionStream error: %+v", err)
	}

	var (
		content     strings.Builder
		reasoning   strings.Builder
		details     []llm.ReasoningDetail
		toolID      string
		toolName    string
		toolIndexes = map[int]bool{}
		toolArgs    strings.Builder
		complete    llm.StreamChunk
	)
	for chunk := range chunks {
		if chunk.Error() != nil {
			t.Fatalf("stream error: %+v", chunk.Error())
		}
		if chunk.IsComplete() {
			complete = chunk
			continue
		}
		delta := chunk.Delta()
		content.WriteString(delta.Content())
		if rd, ok := delta.(llm.ReasoningStreamDelta); ok {
			reasoning.WriteString(rd.Reasoning())
			details = append(details, rd.ReasoningDetails()...)
		}
		for _, tc := range delta.ToolCalls() {
			toolIndexes[tc.Index()] = true
			if tc.ID() != "" {
				toolID = tc.ID()
			}
			if tc.Name() != "" {
				toolName = tc.Name()
			}
			toolArgs.WriteString(tc.ParametersDelta())
		}
	}

	if content.String() != "Hello, world" {
		t.Errorf("unexpected accumulated content: %q", content.String())
	}
	if reasoning.String() != "Let me think." {
		t.Errorf("unexpected accumulated reasoning: %q", reasoning.String())
	}
	if len(details) != 1 || details[0].Signature != "sig-1" || details[0].Text != "Let me think." || details[0].Index != 0 {
		t.Errorf("the streamed detail must carry the full text with its signature: %+v", details)
	}

	// Replaying what the stream accumulated must yield a complete thinking
	// block: a signature only verifies the text it was computed over.
	_, replayed, err := buildMessages([]llm.Message{
		llm.NewMessage(llm.RoleUser, "Weather in Paris?"),
		llm.NewReasoningToolCallsMessage(reasoning.String(), details,
			llm.NewToolCall(toolID, toolName, toolArgs.String())),
		llm.NewToolMessage(toolID, llm.NewToolResult("Sunny")),
	})
	if err != nil {
		t.Fatalf("could not replay the streamed turn: %+v", err)
	}
	thinking := replayed[1].Content[0].OfThinking
	if thinking == nil || thinking.Thinking != "Let me think." || thinking.Signature != "sig-1" {
		t.Errorf("replayed thinking block is incomplete: %+v", replayed[1].Content[0])
	}
	if toolID != "toolu_1" || toolName != "get_weather" || toolArgs.String() != `{"location":"Paris"}` {
		t.Errorf("unexpected tool call stream: id=%q name=%q args=%q", toolID, toolName, toolArgs.String())
	}
	if len(toolIndexes) != 1 || !toolIndexes[0] {
		t.Errorf("tool call index must count tool calls from 0, got %v", toolIndexes)
	}

	if complete == nil {
		t.Fatal("stream did not emit a complete chunk")
	}
	usage := complete.Usage()
	if usage == nil || usage.PromptTokens() != 160 || usage.CompletionTokens() != 20 {
		t.Errorf("unexpected streamed usage: %v", usage)
	}
	if cw, ok := usage.(llm.CacheCreationReportingUsage); !ok || cw.CacheCreationTokens() != 50 {
		t.Errorf("cache writes not reported in stream: %v", usage)
	}
}

func TestChatCompletionStream_Error(t *testing.T) {
	fake := &fakeMessagesAPI{status: http.StatusInternalServerError, errBody: `{"type":"error","error":{"type":"api_error","message":"boom"}}`}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatalf("ChatCompletionStream error: %+v", err)
	}

	var streamErr error
	for chunk := range chunks {
		if chunk.Error() != nil {
			streamErr = chunk.Error()
		}
	}
	if streamErr == nil {
		t.Fatal("expected an error chunk")
	}
	var httpErr *llm.HTTPError
	if !errors.As(streamErr, &httpErr) || httpErr.StatusCode != http.StatusInternalServerError {
		t.Errorf("HTTPError not propagated: %v", streamErr)
	}
}

func TestChatCompletionStream_ErrorEventMidStreamIsRetryable(t *testing.T) {
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"error", map[string]any{"type": "error", "error": map[string]any{"type": "overloaded_error", "message": "Overloaded"}}},
	}}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatalf("ChatCompletionStream error: %+v", err)
	}
	var streamErr error
	for chunk := range chunks {
		if chunk.Error() != nil {
			streamErr = chunk.Error()
		}
	}
	if streamErr == nil {
		t.Fatal("expected an error chunk")
	}
	var httpErr *llm.HTTPError
	if !errors.As(streamErr, &httpErr) {
		t.Fatalf("HTTPError not propagated: %v", streamErr)
	}
	if httpErr.StatusCode == http.StatusOK {
		t.Errorf("an error event must not inherit the 200 of the stream: %v", streamErr)
	}
	if !llm.IsRetryable(streamErr) {
		t.Errorf("an overloaded upstream must be retryable: %v", streamErr)
	}
	if !strings.Contains(httpErr.Body, "Overloaded") {
		t.Errorf("error body lost: %v", httpErr.Body)
	}
}

func TestChatCompletion_RateLimitErrorTypeMidStream(t *testing.T) {
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"error", map[string]any{"type": "error", "error": map[string]any{"type": "rate_limit_error", "message": "slow down"}}},
	}}
	client := newTestClient(t, fake, "")

	_, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if !errors.Is(err, llm.ErrRateLimit) {
		t.Errorf("a rate_limit_error event must be reported as ErrRateLimit, got %v", err)
	}
}

func TestChatCompletion_EmptyStreamIsNoMessage(t *testing.T) {
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 0}}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}}
	client := newTestClient(t, fake, "")

	_, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if !errors.Is(err, llm.ErrNoMessage) {
		t.Errorf("a stream without content must be ErrNoMessage, got %v", err)
	}
}

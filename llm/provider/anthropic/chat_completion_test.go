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
	"time"

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
	rr, ok := res.(llm.ReasoningChatCompletionResponse)
	if !ok {
		t.Fatal("response does not implement ReasoningChatCompletionResponse")
	}
	if rr.Reasoning() != "" {
		t.Errorf("plaintext reasoning must be withheld when excluded: %q", rr.Reasoning())
	}
	// The signed block must stay on the message: the next turn replays it.
	rm, ok := res.Message().(llm.ReasoningMessage)
	if !ok || len(rm.ReasoningDetails()) != 1 || rm.ReasoningDetails()[0].Signature != "sig-1" {
		t.Errorf("signed thinking must be kept for replay even when excluded: %v", res.Message())
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

func TestChatCompletionStream_ExcludeReasoningKeepsSignedBlocks(t *testing.T) {
	fake := &fakeMessagesAPI{events: scriptedReply()}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(),
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithReasoning(&llm.ReasoningOptions{Exclude: true}),
	)
	if err != nil {
		t.Fatal(err)
	}
	var (
		reasoning strings.Builder
		details   []llm.ReasoningDetail
	)
	for chunk := range chunks {
		if chunk.Error() != nil {
			t.Fatal(chunk.Error())
		}
		if rd, ok := chunk.Delta().(llm.ReasoningStreamDelta); ok {
			reasoning.WriteString(rd.Reasoning())
			details = append(details, rd.ReasoningDetails()...)
		}
	}
	if reasoning.String() != "" {
		t.Errorf("incremental reasoning must be silenced when excluded: %q", reasoning.String())
	}
	if len(details) != 1 || details[0].Text != "Let me think." || details[0].Signature != "sig-1" {
		t.Errorf("the signed block must still be emitted for replay: %+v", details)
	}
}

// parallelToolsReply scripts two tool_use blocks plus a redacted thinking
// block, the second tool carrying its whole input on content_block_start
// the way some gateways do.
func parallelToolsReply() []sseEvent {
	return []sseEvent{
		scriptedReply()[0],
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "redacted_thinking", "data": "opaque"}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}},
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 1,
			"content_block": map[string]any{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": map[string]any{}}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 1,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": `{"location":"Paris"}`}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 1}},
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 2,
			"content_block": map[string]any{"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": map[string]any{"location": "London"}}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 2}},
		{"message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 20}}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}
}

func TestChatCompletionStream_ParallelToolCallsAndRedactedThinking(t *testing.T) {
	fake := &fakeMessagesAPI{events: parallelToolsReply()}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatal(err)
	}
	args := map[int]*strings.Builder{}
	ids := map[int]string{}
	var details []llm.ReasoningDetail
	for chunk := range chunks {
		if chunk.Error() != nil {
			t.Fatal(chunk.Error())
		}
		if chunk.IsComplete() {
			continue
		}
		if rd, ok := chunk.Delta().(llm.ReasoningStreamDelta); ok {
			details = append(details, rd.ReasoningDetails()...)
		}
		for _, tc := range chunk.Delta().ToolCalls() {
			if args[tc.Index()] == nil {
				args[tc.Index()] = &strings.Builder{}
			}
			if tc.ID() != "" {
				ids[tc.Index()] = tc.ID()
			}
			args[tc.Index()].WriteString(tc.ParametersDelta())
		}
	}
	if len(details) != 1 || details[0].Type != llm.ReasoningDetailTypeEncrypted || details[0].Data != "opaque" {
		t.Errorf("redacted thinking not streamed: %+v", details)
	}
	if ids[0] != "toolu_1" || ids[1] != "toolu_2" {
		t.Errorf("tool calls must be indexed 0 and 1 in order, got %v", ids)
	}
	if args[0].String() != `{"location":"Paris"}` {
		t.Errorf("unexpected first tool arguments: %q", args[0].String())
	}
	if args[1].String() != `{"location":"London"}` {
		t.Errorf("input carried by content_block_start must be surfaced: %q", args[1].String())
	}
}

func TestChatCompletion_ParallelToolCallsAndRedactedThinking(t *testing.T) {
	fake := &fakeMessagesAPI{events: parallelToolsReply()}
	client := newTestClient(t, fake, "")

	res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls()) != 2 || res.ToolCalls()[1].ID() != "toolu_2" {
		t.Errorf("expected 2 tool calls, got %v", res.ToolCalls())
	}
	if params, _ := res.ToolCalls()[1].Parameters().(string); params != `{"location":"London"}` {
		t.Errorf("tool input must come back as its raw JSON, got %v", res.ToolCalls()[1].Parameters())
	}
	rm, ok := res.Message().(llm.ReasoningMessage)
	if !ok || len(rm.ReasoningDetails()) != 1 || rm.ReasoningDetails()[0].Data != "opaque" {
		t.Errorf("redacted thinking must be kept on the message: %v", res.Message())
	}
	// And it must replay as a redacted_thinking block.
	_, replayed, err := buildMessages([]llm.Message{
		llm.NewMessage(llm.RoleUser, "Hi"),
		llm.NewReasoningToolCallsMessage("", rm.ReasoningDetails(), res.ToolCalls()...),
		llm.NewToolMessage("toolu_1", llm.NewToolResult("Sunny")),
		llm.NewToolMessage("toolu_2", llm.NewToolResult("Rainy")),
	})
	if err != nil {
		t.Fatal(err)
	}
	if replayed[1].Content[0].OfRedactedThinking == nil || len(replayed[2].Content) != 2 {
		t.Errorf("unexpected replay: %+v", replayed)
	}
}

func TestChatCompletion_InvalidRequestMidStreamIsNotRetried(t *testing.T) {
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"error", map[string]any{"type": "error", "error": map[string]any{"type": "invalid_request_error", "message": "bad"}}},
	}}
	client := newTestClient(t, fake, "")

	_, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	var httpErr *llm.HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusBadRequest {
		t.Errorf("an invalid_request_error event must map to 400, got %v", err)
	}
	if llm.IsRetryable(err) {
		t.Error("an invalid request must not be retried")
	}
}

func TestChatCompletionStream_ThinkingCarriedByBlockStart(t *testing.T) {
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "thinking", "thinking": "All at once.", "signature": "sig"}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}},
		{"message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 3}}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatal(err)
	}
	var reasoning strings.Builder
	var details []llm.ReasoningDetail
	for chunk := range chunks {
		if chunk.Error() != nil {
			t.Fatal(chunk.Error())
		}
		if rd, ok := chunk.Delta().(llm.ReasoningStreamDelta); ok {
			reasoning.WriteString(rd.Reasoning())
			details = append(details, rd.ReasoningDetails()...)
		}
	}
	if reasoning.String() != "All at once." {
		t.Errorf("thinking text carried by content_block_start must be streamed: %q", reasoning.String())
	}
	if len(details) != 1 || details[0].Text != "All at once." || details[0].Signature != "sig" {
		t.Errorf("unexpected closing detail: %+v", details)
	}
}

func TestChatCompletion_MalformedStreamIsRetryable(t *testing.T) {
	// A content block starting at index 1 with no block 0 is rejected by
	// Accumulate; the failure must surface as an upstream error.
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 1,
			"content_block": map[string]any{"type": "text", "text": ""}}},
	}}
	client := newTestClient(t, fake, "")

	_, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err == nil {
		t.Fatal("expected an error")
	}
	var httpErr *llm.HTTPError
	if !errors.As(err, &httpErr) || !llm.IsRetryable(err) {
		t.Errorf("a malformed stream must be reported as a retryable upstream failure, got %v", err)
	}
}

func TestChatCompletionStream_EmptyStreamIsNoMessage(t *testing.T) {
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 0}}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatal(err)
	}
	var streamErr error
	var complete bool
	for chunk := range chunks {
		if chunk.Error() != nil {
			streamErr = chunk.Error()
		}
		if chunk.IsComplete() {
			complete = true
		}
	}
	if !errors.Is(streamErr, llm.ErrNoMessage) || complete {
		t.Errorf("an empty stream must end with ErrNoMessage and no complete chunk, got err=%v complete=%v", streamErr, complete)
	}
}

func TestChatCompletionStream_WholeToolInputOnStartIsNotDoubledByDeltas(t *testing.T) {
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": "Who"}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "text_delta", "text": "le."}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}},
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 1,
			"content_block": map[string]any{"type": "tool_use", "id": "toolu_1", "name": "t", "input": map[string]any{"a": 1}}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 1,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": `{"a":1}`}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 1}},
		{"message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 4}}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatal(err)
	}
	var content, args strings.Builder
	for chunk := range chunks {
		if chunk.Error() != nil {
			t.Fatal(chunk.Error())
		}
		if chunk.IsComplete() {
			continue
		}
		content.WriteString(chunk.Delta().Content())
		for _, tc := range chunk.Delta().ToolCalls() {
			args.WriteString(tc.ParametersDelta())
		}
	}
	if content.String() != "Whole." {
		t.Errorf("text opened with its beginning must keep its deltas: %q", content.String())
	}
	if args.String() != `{"a":1}` {
		t.Errorf("tool input doubled: %q", args.String())
	}
}

func TestChatCompletionStream_ThinkingTextOnStartThenSignatureDelta(t *testing.T) {
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "thinking", "thinking": "Whole thought.", "signature": ""}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "signature_delta", "signature": "sig-late"}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}},
		{"message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 3}}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatal(err)
	}
	var details []llm.ReasoningDetail
	for chunk := range chunks {
		if chunk.Error() != nil {
			t.Fatal(chunk.Error())
		}
		if rd, ok := chunk.Delta().(llm.ReasoningStreamDelta); ok {
			details = append(details, rd.ReasoningDetails()...)
		}
	}
	if len(details) != 1 || details[0].Signature != "sig-late" || details[0].Text != "Whole thought." {
		t.Fatalf("a signature arriving after a whole-text start must be kept: %+v", details)
	}
	_, replayed, err := buildMessages([]llm.Message{
		llm.NewMessage(llm.RoleUser, "Hi"),
		llm.NewAssistantReasoningMessage("ok", "", details),
		llm.NewMessage(llm.RoleUser, "Go on."),
	})
	if err != nil {
		t.Fatal(err)
	}
	if replayed[1].Content[0].OfThinking == nil || replayed[1].Content[0].OfThinking.Signature != "sig-late" {
		t.Errorf("the streamed block must replay signed: %+v", replayed[1].Content)
	}
}

func TestChatCompletionStream_TimeoutMidStreamIsRetryable(t *testing.T) {
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"error", map[string]any{"type": "error", "error": map[string]any{"type": "timeout_error", "message": "too slow"}}},
	}}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatal(err)
	}
	var streamErr error
	for chunk := range chunks {
		if chunk.Error() != nil {
			streamErr = chunk.Error()
		}
	}
	if streamErr == nil || !llm.IsRetryable(streamErr) {
		t.Errorf("a timeout_error event must be retryable, got %v", streamErr)
	}
}

func TestChatCompletionStream_EmptyTextBlockMatchesNonStreamed(t *testing.T) {
	events := []sseEvent{
		scriptedReply()[0],
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}},
		{"message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 1}}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}
	client := newTestClient(t, &fakeMessagesAPI{events: events}, "")

	res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil || res.Message().Content() != "" {
		t.Fatalf("non-streamed: expected an empty response without error, got %v / %v", res, err)
	}

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatal(err)
	}
	var complete bool
	for chunk := range chunks {
		if chunk.Error() != nil {
			t.Fatalf("streamed: the same response must not be an error: %v", chunk.Error())
		}
		if chunk.IsComplete() {
			complete = true
		}
	}
	if !complete {
		t.Error("streamed: expected a complete chunk")
	}
}

// emptyInputToolReply scripts a parameterless tool call the way the API
// streams it: an empty object on the start event and no input delta.
func emptyInputToolReply() []sseEvent {
	return []sseEvent{
		scriptedReply()[0],
		{"content_block_start", map[string]any{"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "tool_use", "id": "toolu_1", "name": "todo_read", "input": map[string]any{}}}},
		{"content_block_delta", map[string]any{"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": ""}}},
		{"content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}},
		{"message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 2}}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}
}

// A parameterless tool call must be executable whichever entry point
// produced it: llm.NewToolCall turns empty parameters into "{}", which is
// what both paths rely on.
func TestParameterlessToolCall_BothPaths(t *testing.T) {
	client := newTestClient(t, &fakeMessagesAPI{events: emptyInputToolReply()}, "")
	tool := llm.NewFuncTool("todo_read", "", map[string]any{"type": "object"},
		func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			return llm.NewToolResult("ok"), nil
		})

	res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls()) != 1 || res.ToolCalls()[0].Parameters() != "{}" {
		t.Fatalf("non-streamed: expected {} parameters, got %v", res.ToolCalls())
	}

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatal(err)
	}
	var id, name string
	var params strings.Builder
	for chunk := range chunks {
		if chunk.Error() != nil {
			t.Fatal(chunk.Error())
		}
		if chunk.IsComplete() {
			continue
		}
		for _, tc := range chunk.Delta().ToolCalls() {
			if tc.ID() != "" {
				id, name = tc.ID(), tc.Name()
			}
			params.WriteString(tc.ParametersDelta())
		}
	}
	// The accumulated deltas are empty; the loop rebuilds the call as
	// agent/loop does, and the tool must run.
	streamed := llm.NewToolCall(id, name, params.String())
	if streamed.Parameters() != "{}" {
		t.Fatalf("streamed: expected {} parameters, got %q", streamed.Parameters())
	}
	for _, call := range []llm.ToolCall{res.ToolCalls()[0], streamed} {
		result, err := llm.ExecuteToolCall(context.Background(), call, tool)
		if err != nil || result.Content() != "ok" {
			t.Errorf("tool call %q must execute: %v / %v", call.ID(), result, err)
		}
	}
}

// TestChatCompletionStream_DeltasCarryUsageAsItGoes asserts that every delta
// carries the counters the provider has already published. A consumer whose
// stream is cut short before message_delta — a proxy whose client hangs up, an
// agent loop giving up — keeps the input tokens that were billed all the same,
// instead of recording a request at zero tokens.
func TestChatCompletionStream_DeltasCarryUsageAsItGoes(t *testing.T) {
	fake := &fakeMessagesAPI{events: scriptedReply()}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(),
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Weather in Paris?")),
	)
	if err != nil {
		t.Fatalf("ChatCompletionStream error: %+v", err)
	}

	deltas := 0
	withUsage := 0
	for chunk := range chunks {
		if chunk.Error() != nil {
			t.Fatalf("stream error: %+v", chunk.Error())
		}
		if chunk.IsComplete() {
			continue
		}
		deltas++
		usage := chunk.Usage()
		if usage == nil {
			continue
		}
		withUsage++
		// message_start reports 10 input tokens, 100 cache reads and 50 cache
		// creation, which newUsage folds into the prompt tokens.
		if got, want := usage.PromptTokens(), int64(160); got != want {
			t.Errorf("delta usage PromptTokens() = %d, want %d from message_start", got, want)
		}
	}

	if deltas == 0 {
		t.Fatal("no delta chunk was emitted")
	}
	if withUsage != deltas {
		t.Errorf("%d of %d deltas carried usage, want all of them: message_start published the counters before the first one", withUsage, deltas)
	}
}

// TestChatCompletionStream_ErrorChunkCarriesUsage asserts that a stream dying
// mid-flight still reports what the provider had billed by then.
func TestChatCompletionStream_ErrorChunkCarriesUsage(t *testing.T) {
	fake := &fakeMessagesAPI{events: []sseEvent{
		scriptedReply()[0],
		{"error", map[string]any{"type": "error", "error": map[string]any{"type": "overloaded_error", "message": "Overloaded"}}},
	}}
	client := newTestClient(t, fake, "")

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatalf("ChatCompletionStream error: %+v", err)
	}

	var errChunk llm.StreamChunk
	for chunk := range chunks {
		if chunk.Error() != nil {
			errChunk = chunk
		}
	}
	if errChunk == nil {
		t.Fatal("expected an error chunk")
	}
	usage := errChunk.Usage()
	if usage == nil {
		t.Fatal("error chunk carries no usage: the input tokens billed before the failure are lost")
	}
	if got, want := usage.PromptTokens(), int64(160); got != want {
		t.Errorf("error chunk usage PromptTokens() = %d, want %d", got, want)
	}
}

// TestChatCompletionStream_StopsWhenConsumerGivesUp asserts that a consumer
// abandoning the channel releases the provider goroutine instead of stranding
// it — and the upstream HTTP response with it — on a send nobody reads.
func TestChatCompletionStream_StopsWhenConsumerGivesUp(t *testing.T) {
	fake := &fakeMessagesAPI{events: scriptedReply()}
	client := newTestClient(t, fake, "")

	ctx, cancel := context.WithCancel(context.Background())
	chunks, err := client.ChatCompletionStream(ctx, llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatalf("ChatCompletionStream error: %+v", err)
	}

	// Read one chunk, then walk away without draining the rest.
	<-chunks
	cancel()

	closed := make(chan struct{})
	go func() {
		defer close(closed)
		for range chunks { //nolint:revive // draining, the values are of no use
		}
	}()
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("the provider goroutine never closed its channel after the consumer gave up")
	}
}

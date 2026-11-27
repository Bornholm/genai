package proxy

import (
	"encoding/json"
	"errors"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

func TestEffectiveFinishReason(t *testing.T) {
	cases := []struct {
		reported  llm.FinishReason
		toolCalls bool
		want      llm.FinishReason
	}{
		{"", false, llm.FinishReasonStop},
		{"", true, llm.FinishReasonToolCalls},
		{llm.FinishReasonStop, false, llm.FinishReasonStop},
		// Gemini reports STOP when it asks for tools; an OpenAI client only
		// runs the tools of a turn that ended on tool_calls.
		{llm.FinishReasonStop, true, llm.FinishReasonToolCalls},
		{llm.FinishReasonLength, false, llm.FinishReasonLength},
		// Calls cut by the token limit may be truncated: the cut wins.
		{llm.FinishReasonLength, true, llm.FinishReasonLength},
		{llm.FinishReasonContentFilter, true, llm.FinishReasonContentFilter},
		{llm.FinishReasonToolCalls, false, llm.FinishReasonToolCalls},
	}
	for _, c := range cases {
		if got := effectiveFinishReason(c.reported, c.toolCalls); got != c.want {
			t.Errorf("effectiveFinishReason(%q, %v) = %q, want %q", c.reported, c.toolCalls, got, c.want)
		}
	}
}

// TestFinishReason_ChatCompletion asserts that an answer cut by the token
// limit reaches the client as such, where the proxy used to call every answer
// without tool calls a stop.
func TestFinishReason_ChatCompletion(t *testing.T) {
	client := &mockChatClient{response: llm.NewChatCompletionResponse(
		llm.NewMessage(llm.RoleAssistant, "Hel"), llm.NewChatCompletionUsage(3, 1, 4),
	).WithFinishReason(llm.FinishReasonLength)}
	capture := &capturingPostHook{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "gpt-4"}), WithHook(capture))

	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildChatRequest(t, `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))

	var body struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decoding response: %v\n%s", err, w.Body.String())
	}
	if len(body.Choices) != 1 || body.Choices[0].FinishReason != "length" {
		t.Errorf("finish_reason on the wire = %+v, want length", body.Choices)
	}
	if capture.res == nil || capture.res.FinishReason != llm.FinishReasonLength {
		t.Errorf("ProxyResponse.FinishReason = %v, want length", capture.res)
	}
}

// TestFinishReason_Stream asserts that a reason reported before the
// completion chunk, as OpenAI does ahead of its usage chunk, reaches both the
// client and the hooks.
func TestFinishReason_Stream(t *testing.T) {
	client := &mockStreamingChatClient{chunks: []llm.StreamChunk{
		llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "Hel")),
		llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "")).WithFinishReason(llm.FinishReasonLength),
		llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(3, 1, 4)),
	}}
	capture := &capturingPostHook{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "gpt-4"}), WithHook(capture))

	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildChatRequest(t, `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`))

	if !strings.Contains(w.Body.String(), `"finish_reason":"length"`) {
		t.Errorf("the terminal chunk does not carry finish_reason length:\n%s", w.Body.String())
	}
	if capture.res == nil || capture.res.FinishReason != llm.FinishReasonLength {
		t.Errorf("ProxyResponse.FinishReason = %v, want length", capture.res)
	}
}

// TestFinishReason_InferredWhenUnreported asserts that a provider reporting no
// reason keeps the reason the proxy inferred, on the wire and for the hooks.
func TestFinishReason_InferredWhenUnreported(t *testing.T) {
	client := &mockStreamingChatClient{chunks: []llm.StreamChunk{
		llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "", llm.NewToolCallDelta(0, "call_1", "lookup", "{}"))),
		llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(3, 1, 4)),
	}}
	capture := &capturingPostHook{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "gpt-4"}), WithHook(capture))

	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildChatRequest(t, `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`))

	if !strings.Contains(w.Body.String(), `"finish_reason":"tool_calls"`) {
		t.Errorf("the terminal chunk does not carry finish_reason tool_calls:\n%s", w.Body.String())
	}
	if capture.res == nil || capture.res.FinishReason != llm.FinishReasonToolCalls {
		t.Errorf("ProxyResponse.FinishReason = %v, want tool_calls", capture.res)
	}
}

// TestFinishReason_EmptyOnInterruptedStream asserts that a stream the provider
// cut reports no reason: the client was never told how it ended.
func TestFinishReason_EmptyOnInterruptedStream(t *testing.T) {
	client := &erroringStreamClient{okChunks: 2, err: errors.New("upstream exploded")}
	capture := &capturingPostHook{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "gpt-4"}), WithHook(capture))

	server.handleChatCompletions(httptest.NewRecorder(), buildChatRequest(t, `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`))

	if capture.res == nil {
		t.Fatal("post-response hook was not run")
	}
	if capture.res.FinishReason != "" {
		t.Errorf("ProxyResponse.FinishReason = %q, want empty on an interrupted stream", capture.res.FinishReason)
	}
}

// TestFinishReason_Messages asserts that the Messages API translates the reason
// to its own stop_reason vocabulary, on both paths.
func TestFinishReason_Messages(t *testing.T) {
	t.Run("completion", func(t *testing.T) {
		client := &mockChatClient{response: llm.NewChatCompletionResponse(
			llm.NewMessage(llm.RoleAssistant, "Hel"), llm.NewChatCompletionUsage(3, 1, 4),
		).WithFinishReason(llm.FinishReasonLength)}
		server := NewServer(WithHook(&resolverHook{client: client, model: "claude"}))

		w := httptest.NewRecorder()
		server.handleMessages(w, buildMessagesRequest(t, "/v1/messages", `{"model":"claude","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}`))

		var body struct {
			StopReason string `json:"stop_reason"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Fatalf("decoding response: %v\n%s", err, w.Body.String())
		}
		if body.StopReason != "max_tokens" {
			t.Errorf("stop_reason = %q, want max_tokens", body.StopReason)
		}
	})

	t.Run("stream", func(t *testing.T) {
		client := &mockStreamingChatClient{chunks: []llm.StreamChunk{
			llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "Hel")),
			llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(3, 1, 4)).WithFinishReason(llm.FinishReasonLength),
		}}
		server := NewServer(WithHook(&resolverHook{client: client, model: "claude"}))

		w := httptest.NewRecorder()
		server.handleMessages(w, buildMessagesRequest(t, "/v1/messages", `{"model":"claude","max_tokens":1,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))

		if !strings.Contains(w.Body.String(), `"stop_reason":"max_tokens"`) {
			t.Errorf("message_delta does not carry stop_reason max_tokens:\n%s", w.Body.String())
		}
	})
}

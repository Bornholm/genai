package proxy

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

// A model often narrates before calling a tool. Dropping that text makes the
// replayed history a series of unexplained calls, which pushes the model to
// re-derive its plan and repeat calls it already made.
func TestConvertMessagesKeepsAssistantContentWithToolCalls(t *testing.T) {
	msgs, err := ConvertOpenAIMessagesJSON(json.RawMessage(`[
		{
			"role": "assistant",
			"content": "Let me read the config first.",
			"tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"/etc/hosts\"}"}}
			]
		}
	]`))
	if err != nil {
		t.Fatalf("could not convert messages: %v", err)
	}

	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if got, want := msgs[0].Content(), "Let me read the config first."; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}

	tcm, ok := msgs[0].(llm.ToolCallsMessage)
	if !ok {
		t.Fatalf("expected a ToolCallsMessage, got %T", msgs[0])
	}
	if len(tcm.ToolCalls()) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(tcm.ToolCalls()))
	}
}

// Clients built on the Vercel AI SDK replay reasoning through reasoning_details
// rather than reasoning_content; the proxy must accept it on the way in.
func TestConvertMessagesReadsReasoningDetails(t *testing.T) {
	msgs, err := ConvertOpenAIMessagesJSON(json.RawMessage(`[
		{
			"role": "assistant",
			"content": "",
			"reasoning_details": [
				{"id": "r1", "type": "reasoning.text", "text": "the user wants X", "format": "minimax", "index": 0}
			],
			"tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
			]
		}
	]`))
	if err != nil {
		t.Fatalf("could not convert messages: %v", err)
	}

	rm, ok := msgs[0].(llm.ReasoningMessage)
	if !ok {
		t.Fatalf("expected a ReasoningMessage, got %T", msgs[0])
	}
	details := rm.ReasoningDetails()
	if len(details) != 1 {
		t.Fatalf("expected 1 reasoning detail, got %d", len(details))
	}
	if details[0].Text != "the user wants X" {
		t.Errorf("reasoning text = %q", details[0].Text)
	}
}

func TestFormatChatCompletionResponseKeepsContentWithToolCalls(t *testing.T) {
	res := llm.NewChatCompletionResponse(
		llm.NewToolCallsMessageWithContent("Checking the config."),
		llm.NewChatCompletionUsage(1, 2, 3),
		llm.NewToolCall("call_1", "read_file", `{"path":"/etc/hosts"}`),
	)

	raw, err := json.Marshal(FormatChatCompletionResponse(res, "m"))
	if err != nil {
		t.Fatalf("could not marshal response: %v", err)
	}

	var out struct {
		Choices []struct {
			Message      openAIMessage `json:"message"`
			FinishReason string        `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("could not unmarshal response: %v", err)
	}

	if got := out.Choices[0].Message.Content; got != "Checking the config." {
		t.Errorf("content = %v, want the assistant text", got)
	}
	if got := out.Choices[0].FinishReason; got != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls", got)
	}
}

// Parameters the request struct does not model must reach the provider instead
// of being silently dropped.
func TestParseChatCompletionRequestForwardsUnmappedFields(t *testing.T) {
	_, _, optFuncs, err := ParseChatCompletionRequest(json.RawMessage(`{
		"model": "m",
		"messages": [{"role": "user", "content": "hi"}],
		"temperature": 0.2,
		"stream": true,
		"n": 1,
		"stream_options": {"include_usage": false},
		"parallel_tool_calls": false,
		"top_p": 0.9,
		"stop": ["\n\n"]
	}`))
	if err != nil {
		t.Fatalf("could not parse request: %v", err)
	}

	extra := llm.NewChatCompletionOptions(optFuncs...).ExtraFields

	if got, ok := extra["parallel_tool_calls"]; !ok || got != false {
		t.Errorf("parallel_tool_calls = %v (present: %v), want false", got, ok)
	}
	if got, ok := extra["top_p"]; !ok || got != 0.9 {
		t.Errorf("top_p = %v (present: %v), want 0.9", got, ok)
	}
	if _, ok := extra["stop"]; !ok {
		t.Error("stop should be forwarded")
	}

	// Explicitly mapped or provider-owned fields must not be duplicated.
	for _, k := range []string{"model", "messages", "temperature", "stream", "n", "stream_options"} {
		if _, ok := extra[k]; ok {
			t.Errorf("%q should not be forwarded as an extra field", k)
		}
	}
}

// The object form of tool_choice has no llm equivalent: it must still constrain
// the model and reach the provider verbatim.
func TestParseChatCompletionRequestObjectToolChoice(t *testing.T) {
	_, _, optFuncs, err := ParseChatCompletionRequest(json.RawMessage(`{
		"model": "m",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"type": "function", "function": {"name": "ls", "parameters": {}}}],
		"tool_choice": {"type": "function", "function": {"name": "ls"}}
	}`))
	if err != nil {
		t.Fatalf("could not parse request: %v", err)
	}

	opts := llm.NewChatCompletionOptions(optFuncs...)
	if opts.ToolChoice != llm.ToolChoiceRequired {
		t.Errorf("tool choice = %q, want required", opts.ToolChoice)
	}
	if _, ok := opts.ExtraFields["tool_choice"]; !ok {
		t.Error("object tool_choice should be forwarded verbatim")
	}
}

func TestParseChatCompletionRequestStringToolChoiceNotForwarded(t *testing.T) {
	_, _, optFuncs, err := ParseChatCompletionRequest(json.RawMessage(`{
		"model": "m",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"type": "function", "function": {"name": "ls", "parameters": {}}}],
		"tool_choice": "auto"
	}`))
	if err != nil {
		t.Fatalf("could not parse request: %v", err)
	}

	opts := llm.NewChatCompletionOptions(optFuncs...)
	if opts.ToolChoice != llm.ToolChoiceAuto {
		t.Errorf("tool choice = %q, want auto", opts.ToolChoice)
	}
	if _, ok := opts.ExtraFields["tool_choice"]; ok {
		t.Error("string tool_choice is already mapped and must not be duplicated")
	}
}

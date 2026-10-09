package proxy

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

func TestParseChatCompletionRequest_Basic(t *testing.T) {
	body := json.RawMessage(`{
		"model": "gpt-4",
		"messages": [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": "Hello"}
		],
		"temperature": 0.7,
		"max_tokens": 100
	}`)

	model, stream, opts, err := ParseChatCompletionRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "gpt-4" {
		t.Errorf("model = %q, want %q", model, "gpt-4")
	}
	if stream {
		t.Error("stream should be false")
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if len(compiled.Messages) != 2 {
		t.Errorf("messages = %d, want 2", len(compiled.Messages))
	}
	if compiled.Messages[0].Role() != llm.RoleSystem {
		t.Errorf("first message role = %q, want system", compiled.Messages[0].Role())
	}
	if compiled.Messages[1].Content() != "Hello" {
		t.Errorf("second message content = %q, want Hello", compiled.Messages[1].Content())
	}
}

func TestConvertOpenAIMessagesJSON(t *testing.T) {
	messagesJSON := json.RawMessage(`[
		{"role": "system", "content": "You are helpful."},
		{"role": "user", "content": "Hello"}
	]`)

	msgs, err := ConvertOpenAIMessagesJSON(messagesJSON)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want 2", len(msgs))
	}
	if msgs[0].Role() != llm.RoleSystem {
		t.Errorf("first message role = %q, want system", msgs[0].Role())
	}
	if msgs[1].Content() != "Hello" {
		t.Errorf("second message content = %q, want Hello", msgs[1].Content())
	}
}

func TestConvertOpenAIMessagesJSON_InvalidJSON(t *testing.T) {
	if _, err := ConvertOpenAIMessagesJSON(json.RawMessage(`not json`)); err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseChatCompletionRequest_Stream(t *testing.T) {
	body := json.RawMessage(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	_, stream, _, err := ParseChatCompletionRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !stream {
		t.Error("stream should be true")
	}
}

func TestParseChatCompletionRequest_ToolMessage(t *testing.T) {
	body := json.RawMessage(`{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "what is 2+2?"},
			{"role": "tool", "content": "4", "tool_call_id": "call_123"}
		]
	}`)

	_, _, opts, err := ParseChatCompletionRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if len(compiled.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(compiled.Messages))
	}
	toolMsg, ok := compiled.Messages[1].(llm.ToolMessage)
	if !ok {
		t.Fatal("second message should implement ToolMessage")
	}
	if toolMsg.ID() != "call_123" {
		t.Errorf("tool message ID = %q, want call_123", toolMsg.ID())
	}
}

func TestParseChatCompletionRequest_InvalidJSON(t *testing.T) {
	_, _, _, err := ParseChatCompletionRequest(json.RawMessage(`not json`))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

// TestParseChatCompletionRequest_ToolChoiceDefaultsToAuto verifies that when
// tools are provided without an explicit tool_choice, the model is still
// allowed to call them. llm.NewChatCompletionOptions defaults ToolChoice to
// "none", which would otherwise silently disable tool calling.
func TestParseChatCompletionRequest_ToolChoiceDefaultsToAuto(t *testing.T) {
	body := json.RawMessage(`{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "Hi"}],
		"tools": [{"type": "function", "function": {"name": "calculator", "description": "evaluates expressions", "parameters": {"type": "object"}}}]
	}`)

	_, _, opts, err := ParseChatCompletionRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if compiled.ToolChoice != llm.ToolChoiceAuto {
		t.Errorf("tool choice = %q, want %q", compiled.ToolChoice, llm.ToolChoiceAuto)
	}
}

// TestParseChatCompletionRequest_NoToolsKeepsDefaultToolChoice verifies that
// without any tools, the parser forces no tool choice: the value seen
// downstream is whatever llm.NewChatCompletionOptions defaults to.
//
// That default is now ToolChoiceAuto. It used to be ToolChoiceNone, which
// silently prevented models from ever calling a tool when a caller relied on
// the default — the reason it was changed.
func TestParseChatCompletionRequest_NoToolsKeepsDefaultToolChoice(t *testing.T) {
	body := json.RawMessage(`{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "Hi"}]
	}`)

	_, _, opts, err := ParseChatCompletionRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if compiled.ToolChoice != llm.NewChatCompletionOptions().ToolChoice {
		t.Errorf("tool choice = %q, want the library default %q", compiled.ToolChoice, llm.NewChatCompletionOptions().ToolChoice)
	}
}

func TestParseEmbeddingRequest_StringInput(t *testing.T) {
	body := json.RawMessage(`{"model":"text-embedding-ada-002","input":"hello world"}`)
	model, inputs, _, err := ParseEmbeddingRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "text-embedding-ada-002" {
		t.Errorf("model = %q", model)
	}
	if len(inputs) != 1 || inputs[0] != "hello world" {
		t.Errorf("inputs = %v", inputs)
	}
}

func TestParseEmbeddingRequest_ArrayInput(t *testing.T) {
	body := json.RawMessage(`{"model":"m","input":["a","b","c"]}`)
	_, inputs, _, err := ParseEmbeddingRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(inputs) != 3 {
		t.Errorf("inputs len = %d, want 3", len(inputs))
	}
}

func TestFormatChatCompletionResponse(t *testing.T) {
	msg := llm.NewMessage(llm.RoleAssistant, "Hello!")
	usage := llm.NewChatCompletionUsage(10, 5, 15)
	res := llm.NewChatCompletionResponse(msg, usage)

	body := FormatChatCompletionResponse(res, "gpt-4")
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if m["object"] != "chat.completion" {
		t.Errorf("object = %v", m["object"])
	}
	if m["model"] != "gpt-4" {
		t.Errorf("model = %v", m["model"])
	}

	choices, ok := m["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatal("no choices in response")
	}
	choice := choices[0].(map[string]any)
	message := choice["message"].(map[string]any)
	if message["content"] != "Hello!" {
		t.Errorf("content = %v", message["content"])
	}
}

func TestFormatModelsResponse(t *testing.T) {
	models := []ModelInfo{
		{ID: "gpt-4", OwnedBy: "proxy"},
	}
	body := FormatModelsResponse(models)
	raw, _ := json.Marshal(body)
	var m map[string]any
	_ = json.Unmarshal(raw, &m)

	if m["object"] != "list" {
		t.Errorf("object = %v", m["object"])
	}
	data := m["data"].([]any)
	if len(data) != 1 {
		t.Fatalf("data len = %d, want 1", len(data))
	}
	entry := data[0].(map[string]any)
	if entry["id"] != "gpt-4" {
		t.Errorf("id = %v", entry["id"])
	}
}

func TestConvertOpenAIMessagesJSON_DeveloperRole(t *testing.T) {
	messagesJSON := json.RawMessage(`[
		{"role": "developer", "content": "You are helpful."},
		{"role": "user", "content": "Hello"}
	]`)

	msgs, err := ConvertOpenAIMessagesJSON(messagesJSON)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want 2", len(msgs))
	}
	if msgs[0].Role() != llm.RoleSystem {
		t.Errorf("first message role = %q, want system", msgs[0].Role())
	}
	if msgs[0].Content() != "You are helpful." {
		t.Errorf("first message content = %q, want %q", msgs[0].Content(), "You are helpful.")
	}
}

func TestConvertOpenAIMessagesJSON_DeveloperRoleWithContentParts(t *testing.T) {
	messagesJSON := json.RawMessage(`[
		{"role": "developer", "content": [{"type": "text", "text": "Be brief."}]}
	]`)

	msgs, err := ConvertOpenAIMessagesJSON(messagesJSON)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("messages = %d, want 1", len(msgs))
	}
	if msgs[0].Role() != llm.RoleSystem {
		t.Errorf("role = %q, want system", msgs[0].Role())
	}
	if msgs[0].Content() != "Be brief." {
		t.Errorf("content = %q, want %q", msgs[0].Content(), "Be brief.")
	}
}

func convertMessagesJSON(t *testing.T, body string) []llm.Message {
	t.Helper()

	msgs, err := ConvertOpenAIMessagesJSON(json.RawMessage(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return msgs
}

func TestConvertOpenAIMessagesJSON_ToolKeepsCacheControl(t *testing.T) {
	msgs := convertMessagesJSON(t, `[
		{"role": "user", "content": "Run it"},
		{"role": "assistant", "tool_calls": [
			{"id": "call_01", "type": "function", "function": {"name": "run", "arguments": "{}"}}
		]},
		{"role": "tool", "tool_call_id": "call_01", "content": [
			{"type": "text", "text": "done", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
		]}
	]`)

	last := msgs[len(msgs)-1]
	if last.Role() != llm.RoleTool || last.Content() != "done" {
		t.Fatalf("last message = %s %q, want the tool result", last.Role(), last.Content())
	}
	cc := cacheControlOf(last)
	if cc == nil || cc.Type != "ephemeral" || cc.TTL == nil || *cc.TTL != "1h" {
		t.Errorf("tool cache control = %+v, want ephemeral with ttl 1h", cc)
	}
}

func TestConvertOpenAIMessagesJSON_AssistantKeepsCacheControl(t *testing.T) {
	marked := `[{"type": "text", "text": "Running.", "cache_control": {"type": "ephemeral"}}]`

	for name, tc := range map[string]struct {
		message string
		role    llm.Role
	}{
		"text": {
			message: `{"role": "assistant", "content": ` + marked + `}`,
			role:    llm.RoleAssistant,
		},
		"reasoning": {
			message: `{"role": "assistant", "reasoning_content": "Thinking.", "content": ` + marked + `}`,
			role:    llm.RoleAssistant,
		},
		"tool calls": {
			message: `{"role": "assistant", "content": ` + marked + `, "tool_calls": [
				{"id": "call_01", "type": "function", "function": {"name": "run", "arguments": "{}"}}
			]}`,
			role: llm.RoleToolCalls,
		},
		"reasoning tool calls": {
			message: `{"role": "assistant", "reasoning_content": "Thinking.", "content": ` + marked + `, "tool_calls": [
				{"id": "call_01", "type": "function", "function": {"name": "run", "arguments": "{}"}}
			]}`,
			role: llm.RoleToolCalls,
		},
	} {
		t.Run(name, func(t *testing.T) {
			msgs := convertMessagesJSON(t, `[{"role": "user", "content": "Run it"}, `+tc.message+`]`)

			last := msgs[len(msgs)-1]
			if last.Role() != tc.role {
				t.Fatalf("role = %q, want %q", last.Role(), tc.role)
			}
			if cc := cacheControlOf(last); cc == nil || cc.Type != "ephemeral" {
				t.Errorf("assistant cache control = %+v, want ephemeral", cc)
			}
		})
	}
}

func TestConvertOpenAIMessagesJSON_AttachmentKeepsCacheControl(t *testing.T) {
	msgs := convertMessagesJSON(t, `[
		{"role": "user", "content": [
			{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}},
			{"type": "text", "text": "What is this?", "cache_control": {"type": "ephemeral"}}
		]}
	]`)

	if len(msgs[0].Attachments()) != 1 {
		t.Fatalf("attachments = %d, want 1", len(msgs[0].Attachments()))
	}
	if cc := cacheControlOf(msgs[0]); cc == nil || cc.Type != "ephemeral" {
		t.Errorf("user cache control = %+v, want ephemeral", cc)
	}
}

func TestConvertOpenAIMessagesJSON_UnmarkedMessagesHaveNoCacheControl(t *testing.T) {
	msgs := convertMessagesJSON(t, `[
		{"role": "user", "content": "Run it"},
		{"role": "assistant", "content": "Running.", "tool_calls": [
			{"id": "call_01", "type": "function", "function": {"name": "run", "arguments": "{}"}}
		]},
		{"role": "tool", "tool_call_id": "call_01", "content": [{"type": "text", "text": "done"}]}
	]`)

	for _, m := range msgs {
		if cc := cacheControlOf(m); cc != nil {
			t.Errorf("%s message has unexpected cache control %+v", m.Role(), cc)
		}
	}
}

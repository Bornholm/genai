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

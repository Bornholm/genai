package proxy

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

func TestConvertAnthropicMessagesJSON(t *testing.T) {
	messagesJSON := json.RawMessage(`[
		{"role": "user", "content": "Hello"},
		{"role": "assistant", "content": "Hi there!"}
	]`)

	msgs, err := ConvertAnthropicMessagesJSON(messagesJSON)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want 2", len(msgs))
	}
	if msgs[0].Role() != llm.RoleUser {
		t.Errorf("first message role = %q, want user", msgs[0].Role())
	}
	if msgs[0].Content() != "Hello" {
		t.Errorf("first message content = %q, want Hello", msgs[0].Content())
	}
	if msgs[1].Role() != llm.RoleAssistant {
		t.Errorf("second message role = %q, want assistant", msgs[1].Role())
	}
	if msgs[1].Content() != "Hi there!" {
		t.Errorf("second message content = %q, want Hi there!", msgs[1].Content())
	}
}

func TestConvertAnthropicMessagesJSON_InvalidJSON(t *testing.T) {
	if _, err := ConvertAnthropicMessagesJSON(json.RawMessage(`not json`)); err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseMessagesRequest_Basic(t *testing.T) {
	body := json.RawMessage(`{
		"model": "claude-3-5-sonnet-20241022",
		"max_tokens": 1024,
		"temperature": 0.5,
		"system": "You are a helpful assistant.",
		"messages": [
			{"role": "user", "content": "Hello"}
		]
	}`)

	model, stream, opts, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "claude-3-5-sonnet-20241022" {
		t.Errorf("model = %q", model)
	}
	if stream {
		t.Error("stream should be false")
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if len(compiled.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(compiled.Messages))
	}
	if compiled.Messages[0].Role() != llm.RoleSystem {
		t.Errorf("first message role = %q, want system", compiled.Messages[0].Role())
	}
	if compiled.Messages[0].Content() != "You are a helpful assistant." {
		t.Errorf("system content = %q", compiled.Messages[0].Content())
	}
	if compiled.Messages[1].Role() != llm.RoleUser {
		t.Errorf("second message role = %q, want user", compiled.Messages[1].Role())
	}
	if compiled.Messages[1].Content() != "Hello" {
		t.Errorf("second message content = %q", compiled.Messages[1].Content())
	}
	if compiled.MaxCompletionTokens == nil || *compiled.MaxCompletionTokens != 1024 {
		t.Errorf("max completion tokens = %v, want 1024", compiled.MaxCompletionTokens)
	}
	if compiled.Temperature != 0.5 {
		t.Errorf("temperature = %v, want 0.5", compiled.Temperature)
	}
}

func TestParseMessagesRequest_Stream(t *testing.T) {
	body := json.RawMessage(`{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	_, stream, _, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !stream {
		t.Error("stream should be true")
	}
}

func TestParseMessagesRequest_InvalidJSON(t *testing.T) {
	_, _, _, err := ParseMessagesRequest(json.RawMessage(`not json`))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseMessagesRequest_SystemArrayWithCacheControl(t *testing.T) {
	body := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"system": [
			{"type": "text", "text": "You are Claude.", "cache_control": {"type": "ephemeral"}}
		],
		"messages": [
			{"role": "user", "content": "Hi"}
		]
	}`)

	_, _, opts, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if len(compiled.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(compiled.Messages))
	}

	system := compiled.Messages[0]
	if system.Role() != llm.RoleSystem {
		t.Errorf("first message role = %q, want system", system.Role())
	}
	if system.Content() != "You are Claude." {
		t.Errorf("system content = %q", system.Content())
	}

	ccMsg, ok := system.(llm.CacheControlMessage)
	if !ok {
		t.Fatal("system message should implement CacheControlMessage")
	}
	cc := ccMsg.CacheControl()
	if cc == nil || cc.Type != "ephemeral" {
		t.Errorf("cache control = %+v, want ephemeral", cc)
	}
}

func TestParseMessagesRequest_ToolUseAndResult(t *testing.T) {
	body := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "What is 2+2?"},
			{"role": "assistant", "content": [
				{"type": "text", "text": "Let me calculate that."},
				{"type": "tool_use", "id": "toolu_01", "name": "calculator", "input": {"expression": "2+2"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_01", "content": "4"}
			]}
		]
	}`)

	_, _, opts, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if len(compiled.Messages) != 4 {
		t.Fatalf("messages = %d, want 4: %#v", len(compiled.Messages), compiled.Messages)
	}

	if compiled.Messages[0].Role() != llm.RoleUser || compiled.Messages[0].Content() != "What is 2+2?" {
		t.Errorf("messages[0] = %+v", compiled.Messages[0])
	}

	if compiled.Messages[1].Role() != llm.RoleAssistant || compiled.Messages[1].Content() != "Let me calculate that." {
		t.Errorf("messages[1] = %+v", compiled.Messages[1])
	}

	toolCallsMsg, ok := compiled.Messages[2].(llm.ToolCallsMessage)
	if !ok {
		t.Fatalf("messages[2] should implement ToolCallsMessage, got %T", compiled.Messages[2])
	}
	calls := toolCallsMsg.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(calls))
	}
	if calls[0].ID() != "toolu_01" {
		t.Errorf("tool call id = %q, want toolu_01", calls[0].ID())
	}
	if calls[0].Name() != "calculator" {
		t.Errorf("tool call name = %q, want calculator", calls[0].Name())
	}

	var params map[string]any
	paramsStr, ok := calls[0].Parameters().(string)
	if !ok {
		t.Fatalf("tool call parameters should be a string, got %T", calls[0].Parameters())
	}
	if err := json.Unmarshal([]byte(paramsStr), &params); err != nil {
		t.Fatalf("could not unmarshal tool call parameters: %v", err)
	}
	if params["expression"] != "2+2" {
		t.Errorf("tool call params = %+v", params)
	}

	toolMsg, ok := compiled.Messages[3].(llm.ToolMessage)
	if !ok {
		t.Fatalf("messages[3] should implement ToolMessage, got %T", compiled.Messages[3])
	}
	if toolMsg.ID() != "toolu_01" {
		t.Errorf("tool message id = %q, want toolu_01", toolMsg.ID())
	}
	if toolMsg.Content() != "4" {
		t.Errorf("tool message content = %q, want 4", toolMsg.Content())
	}
}

func TestParseMessagesRequest_ToolResultIsError(t *testing.T) {
	body := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_01", "content": "boom", "is_error": true}
			]}
		]
	}`)

	_, _, opts, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if len(compiled.Messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(compiled.Messages))
	}
	toolMsg, ok := compiled.Messages[0].(llm.ToolMessage)
	if !ok {
		t.Fatalf("messages[0] should implement ToolMessage, got %T", compiled.Messages[0])
	}
	if toolMsg.Content() != "Error: boom" {
		t.Errorf("tool message content = %q, want %q", toolMsg.Content(), "Error: boom")
	}
}

func TestParseMessagesRequest_Image(t *testing.T) {
	body := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": [
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
				{"type": "text", "text": "What is in this image?"}
			]}
		]
	}`)

	_, _, opts, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if len(compiled.Messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(compiled.Messages))
	}

	msg := compiled.Messages[0]
	if msg.Content() != "What is in this image?" {
		t.Errorf("content = %q", msg.Content())
	}

	attachments := msg.Attachments()
	if len(attachments) != 1 {
		t.Fatalf("attachments = %d, want 1", len(attachments))
	}
	if attachments[0].Type() != llm.AttachmentTypeImage {
		t.Errorf("attachment type = %q, want image", attachments[0].Type())
	}
	if attachments[0].MimeType() != "image/png" {
		t.Errorf("attachment mime type = %q, want image/png", attachments[0].MimeType())
	}
}

func TestParseMessagesRequest_Thinking(t *testing.T) {
	body := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"thinking": {"type": "enabled", "budget_tokens": 2000},
		"messages": [
			{"role": "user", "content": "Hi"}
		]
	}`)

	_, _, opts, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if compiled.Reasoning == nil {
		t.Fatal("reasoning options should be set")
	}
	if compiled.Reasoning.MaxTokens == nil || *compiled.Reasoning.MaxTokens != 2000 {
		t.Errorf("reasoning max tokens = %v, want 2000", compiled.Reasoning.MaxTokens)
	}
	if compiled.Reasoning.Enabled == nil || !*compiled.Reasoning.Enabled {
		t.Errorf("reasoning enabled = %v, want true", compiled.Reasoning.Enabled)
	}
}

func TestParseMessagesRequest_ToolChoice(t *testing.T) {
	tests := []struct {
		name       string
		toolChoice string
		want       llm.ToolChoice
	}{
		{"auto", `{"type":"auto"}`, llm.ToolChoiceAuto},
		{"any", `{"type":"any"}`, llm.ToolChoiceRequired},
		{"none", `{"type":"none"}`, llm.ToolChoiceNone},
		{"tool", `{"type":"tool","name":"calculator"}`, llm.ToolChoiceRequired},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := json.RawMessage(`{
				"model": "m",
				"max_tokens": 100,
				"tool_choice": ` + tt.toolChoice + `,
				"tools": [{"name": "calculator", "description": "evaluates expressions", "input_schema": {"type": "object"}}],
				"messages": [{"role": "user", "content": "Hi"}]
			}`)

			_, _, opts, err := ParseMessagesRequest(body)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			compiled := llm.NewChatCompletionOptions(opts...)
			if compiled.ToolChoice != tt.want {
				t.Errorf("tool choice = %q, want %q", compiled.ToolChoice, tt.want)
			}
			if len(compiled.Tools) != 1 || compiled.Tools[0].Name() != "calculator" {
				t.Errorf("tools = %+v", compiled.Tools)
			}
		})
	}
}

// TestParseMessagesRequest_ToolChoiceDefaultsToAuto verifies that when tools
// are provided without an explicit tool_choice, the model is still allowed
// to call them. llm.NewChatCompletionOptions defaults ToolChoice to "none",
// which would otherwise silently disable tool calling.
func TestParseMessagesRequest_ToolChoiceDefaultsToAuto(t *testing.T) {
	body := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"tools": [{"name": "calculator", "description": "evaluates expressions", "input_schema": {"type": "object"}}],
		"messages": [{"role": "user", "content": "Hi"}]
	}`)

	_, _, opts, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if compiled.ToolChoice != llm.ToolChoiceAuto {
		t.Errorf("tool choice = %q, want %q", compiled.ToolChoice, llm.ToolChoiceAuto)
	}
}

// TestParseMessagesRequest_NoToolsToolChoiceNone verifies that without any
// tools, ToolChoice keeps its default value of "none".
func TestParseMessagesRequest_NoToolsToolChoiceNone(t *testing.T) {
	body := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`)

	_, _, opts, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	compiled := llm.NewChatCompletionOptions(opts...)
	if compiled.ToolChoice != llm.ToolChoiceNone {
		t.Errorf("tool choice = %q, want %q", compiled.ToolChoice, llm.ToolChoiceNone)
	}
}

func TestParseMessagesRequest_StopSequencesIgnored(t *testing.T) {
	body := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"stop_sequences": ["STOP"],
		"top_p": 0.9,
		"top_k": 40,
		"messages": [{"role": "user", "content": "Hi"}]
	}`)

	_, _, _, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestFormatMessagesResponse_Text(t *testing.T) {
	msg := llm.NewMessage(llm.RoleAssistant, "Hello!")
	usage := llm.NewChatCompletionUsage(10, 5, 15)
	res := llm.NewChatCompletionResponse(msg, usage)

	body := FormatMessagesResponse(res, "claude-3-5-sonnet-20241022")
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if m["type"] != "message" {
		t.Errorf("type = %v, want message", m["type"])
	}
	if m["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", m["role"])
	}
	if m["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v, want end_turn", m["stop_reason"])
	}

	content := m["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("content = %d, want 1", len(content))
	}
	block := content[0].(map[string]any)
	if block["type"] != "text" || block["text"] != "Hello!" {
		t.Errorf("content[0] = %+v", block)
	}

	usageOut := m["usage"].(map[string]any)
	if usageOut["input_tokens"] != float64(10) || usageOut["output_tokens"] != float64(5) {
		t.Errorf("usage = %+v", usageOut)
	}
}

func TestFormatMessagesResponse_ToolCalls(t *testing.T) {
	msg := llm.NewMessage(llm.RoleAssistant, "")
	usage := llm.NewChatCompletionUsage(10, 5, 15)
	tc := llm.NewToolCall("toolu_1", "get_weather", `{"location":"Paris"}`)
	res := llm.NewChatCompletionResponse(msg, usage, tc)

	body := FormatMessagesResponse(res, "claude-3-5-sonnet-20241022")
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if m["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use", m["stop_reason"])
	}

	content := m["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("content = %d, want 1", len(content))
	}
	block := content[0].(map[string]any)
	if block["type"] != "tool_use" {
		t.Errorf("content[0].type = %v, want tool_use", block["type"])
	}
	if block["id"] != "toolu_1" || block["name"] != "get_weather" {
		t.Errorf("content[0] = %+v", block)
	}
	input, ok := block["input"].(map[string]any)
	if !ok {
		t.Fatalf("content[0].input should be an object, got %T", block["input"])
	}
	if input["location"] != "Paris" {
		t.Errorf("input = %+v", input)
	}
}

func TestFormatMessagesResponse_Reasoning(t *testing.T) {
	msg := llm.NewMessage(llm.RoleAssistant, "The answer is 4.")
	usage := llm.NewChatCompletionUsage(10, 5, 15)
	details := []llm.ReasoningDetail{
		{Type: llm.ReasoningDetailTypeText, Text: "2+2 is 4", Signature: "sig123"},
	}
	res := llm.NewChatCompletionResponseWithReasoning(msg, usage, "2+2 is 4", details)

	body := FormatMessagesResponse(res, "claude-3-5-sonnet-20241022")
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	content := m["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content = %d, want 2: %+v", len(content), content)
	}

	thinking := content[0].(map[string]any)
	if thinking["type"] != "thinking" || thinking["thinking"] != "2+2 is 4" {
		t.Errorf("content[0] = %+v", thinking)
	}
	if thinking["signature"] != "sig123" {
		t.Errorf("content[0].signature = %v, want sig123", thinking["signature"])
	}

	text := content[1].(map[string]any)
	if text["type"] != "text" || text["text"] != "The answer is 4." {
		t.Errorf("content[1] = %+v", text)
	}
}

func TestEstimateTokenCount(t *testing.T) {
	body := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"system": "You are a helpful assistant.",
		"messages": [
			{"role": "user", "content": "Hello, how are you doing today?"}
		]
	}`)

	count := EstimateTokenCount(body)
	if count <= 0 {
		t.Errorf("count = %d, want > 0", count)
	}
	if count > len(body) {
		t.Errorf("count = %d, should be smaller than the raw body length %d", count, len(body))
	}
}

func TestEstimateTokenCount_Image(t *testing.T) {
	withoutImage := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "Hi"}
		]
	}`)
	withImage := json.RawMessage(`{
		"model": "m",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": [
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
				{"type": "text", "text": "Hi"}
			]}
		]
	}`)

	withoutCount := EstimateTokenCount(withoutImage)
	withCount := EstimateTokenCount(withImage)

	if withCount-withoutCount < estimatedTokensPerImage {
		t.Errorf("image should add at least %d tokens: without=%d, with=%d", estimatedTokensPerImage, withoutCount, withCount)
	}
}

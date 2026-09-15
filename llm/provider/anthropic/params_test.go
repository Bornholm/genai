package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

// marshalParams builds the request for the given options and decodes the
// wire JSON, which is what the assertions care about.
func marshalParams(t *testing.T, funcs ...llm.ChatCompletionOptionFunc) map[string]any {
	t.Helper()
	params, err := buildParams(llm.NewChatCompletionOptions(funcs...), "claude-sonnet-5", DefaultMaxTokens)
	if err != nil {
		t.Fatalf("could not build params: %+v", err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("could not marshal params: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("could not decode params: %v", err)
	}
	return body
}

func messagesOf(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	raw, _ := body["messages"].([]any)
	messages := make([]map[string]any, 0, len(raw))
	for _, m := range raw {
		messages = append(messages, m.(map[string]any))
	}
	return messages
}

func blocksOf(t *testing.T, message map[string]any) []map[string]any {
	t.Helper()
	raw, _ := message["content"].([]any)
	blocks := make([]map[string]any, 0, len(raw))
	for _, b := range raw {
		blocks = append(blocks, b.(map[string]any))
	}
	return blocks
}

func TestBuildParams_SystemAndCacheControl(t *testing.T) {
	ttl := "1h"
	body := marshalParams(t, llm.WithMessages(
		llm.NewMessageWithCacheControl(llm.RoleSystem, "You are terse.", &llm.CacheControl{Type: "ephemeral", TTL: &ttl}),
		llm.NewMessageWithCacheControl(llm.RoleUser, "Hello", &llm.CacheControl{Type: "ephemeral"}),
	))

	system, _ := body["system"].([]any)
	if len(system) != 1 {
		t.Fatalf("expected 1 system block, got %v", body["system"])
	}
	sysBlock := system[0].(map[string]any)
	cc, _ := sysBlock["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" || cc["ttl"] != "1h" {
		t.Errorf("unexpected system cache_control: %v", sysBlock["cache_control"])
	}

	messages := messagesOf(t, body)
	if len(messages) != 1 || messages[0]["role"] != "user" {
		t.Fatalf("expected a single user turn, got %v", messages)
	}
	blocks := blocksOf(t, messages[0])
	cc, _ = blocks[0]["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Errorf("user block lost its cache_control: %v", blocks[0])
	}
	if _, present := cc["ttl"]; present {
		t.Errorf("ttl must be omitted when not set, got %v", cc)
	}
}

func TestBuildParams_NoHintKeepsBlocksClean(t *testing.T) {
	body := marshalParams(t, llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hello")))
	blocks := blocksOf(t, messagesOf(t, body)[0])
	if _, present := blocks[0]["cache_control"]; present {
		t.Errorf("cache_control emitted without a hint: %v", blocks[0])
	}
	if body["max_tokens"] != float64(DefaultMaxTokens) {
		t.Errorf("expected default max_tokens %d, got %v", DefaultMaxTokens, body["max_tokens"])
	}
}

func TestBuildParams_FoldsToolResultsIntoOneUserTurn(t *testing.T) {
	first := llm.NewToolCall("call_1", "get_weather", `{"location":"Paris"}`)
	second := llm.NewToolCall("call_2", "get_weather", `{"location":"London"}`)

	body := marshalParams(t, llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Weather in Paris and London?"),
		llm.NewToolCallsMessageWithContent("Let me check.", first, second),
		llm.NewToolMessage("call_1", llm.NewToolResult("Sunny")),
		llm.NewToolMessage("call_2", llm.NewToolResult("Rainy")),
	))

	messages := messagesOf(t, body)
	if len(messages) != 3 {
		t.Fatalf("expected user/assistant/user, got %d turns: %v", len(messages), messages)
	}

	assistant := blocksOf(t, messages[1])
	if messages[1]["role"] != "assistant" || len(assistant) != 3 {
		t.Fatalf("expected text + 2 tool_use blocks, got %v", messages[1])
	}
	if assistant[0]["type"] != "text" || assistant[0]["text"] != "Let me check." {
		t.Errorf("assistant rationale lost: %v", assistant[0])
	}
	if assistant[1]["type"] != "tool_use" || assistant[1]["id"] != "call_1" {
		t.Errorf("unexpected first tool_use: %v", assistant[1])
	}
	input, _ := assistant[1]["input"].(map[string]any)
	if input["location"] != "Paris" {
		t.Errorf("tool_use input not decoded as an object: %v", assistant[1]["input"])
	}

	results := blocksOf(t, messages[2])
	if messages[2]["role"] != "user" || len(results) != 2 {
		t.Fatalf("expected both tool_result blocks in one user turn, got %v", messages[2])
	}
	if results[0]["tool_use_id"] != "call_1" || results[1]["tool_use_id"] != "call_2" {
		t.Errorf("tool_result ids out of order: %v", results)
	}
}

func TestBuildParams_ToolMessageWithoutToolInterfaceIsRejected(t *testing.T) {
	_, err := buildParams(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMessage(llm.RoleTool, "orphan"),
	)), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Fatal("expected an error for a tool message without an ID")
	}
}

func TestBuildParams_ReplaysSignedThinkingOnly(t *testing.T) {
	body := marshalParams(t, llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Think about it."),
		llm.NewAssistantReasoningMessage("42", "raw reasoning", []llm.ReasoningDetail{
			{Type: llm.ReasoningDetailTypeText, Text: "unsigned"},
			{Type: llm.ReasoningDetailTypeText, Text: "signed", Signature: "sig"},
			{Type: llm.ReasoningDetailTypeEncrypted, Data: "opaque"},
		}),
		llm.NewMessage(llm.RoleUser, "Sure?"),
	))

	blocks := blocksOf(t, messagesOf(t, body)[1])
	if len(blocks) != 3 {
		t.Fatalf("expected thinking + redacted_thinking + text, got %v", blocks)
	}
	if blocks[0]["type"] != "thinking" || blocks[0]["signature"] != "sig" || blocks[0]["thinking"] != "signed" {
		t.Errorf("unexpected thinking block: %v", blocks[0])
	}
	if blocks[1]["type"] != "redacted_thinking" || blocks[1]["data"] != "opaque" {
		t.Errorf("unexpected redacted_thinking block: %v", blocks[1])
	}
	if blocks[2]["type"] != "text" || blocks[2]["text"] != "42" {
		t.Errorf("text must come after thinking: %v", blocks[2])
	}
}

func TestBuildParams_Thinking(t *testing.T) {
	t.Run("effort share of max_tokens drops temperature", func(t *testing.T) {
		body := marshalParams(t,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithMaxCompletionTokens(10000),
			llm.WithTemperature(0.2),
			llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortHigh)),
		)
		thinking, _ := body["thinking"].(map[string]any)
		if thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(8000) {
			t.Errorf("unexpected thinking config: %v", body["thinking"])
		}
		if _, present := body["temperature"]; present {
			t.Error("temperature must be dropped when thinking is enabled")
		}
	})

	t.Run("explicit budget raises max_tokens", func(t *testing.T) {
		budget := 6000
		body := marshalParams(t,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithReasoning(&llm.ReasoningOptions{MaxTokens: &budget}),
		)
		thinking, _ := body["thinking"].(map[string]any)
		if thinking["budget_tokens"] != float64(6000) {
			t.Errorf("unexpected budget: %v", body["thinking"])
		}
		if body["max_tokens"] != float64(6000+DefaultMaxTokens) {
			t.Errorf("max_tokens must exceed the budget, got %v", body["max_tokens"])
		}
	})

	t.Run("budget is clamped to the API minimum", func(t *testing.T) {
		body := marshalParams(t,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortMinimal)),
		)
		thinking, _ := body["thinking"].(map[string]any)
		if thinking["budget_tokens"] != float64(minThinkingBudget) {
			t.Errorf("expected minimum budget, got %v", body["thinking"])
		}
	})

	t.Run("none keeps temperature and no thinking", func(t *testing.T) {
		body := marshalParams(t,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithTemperature(0.7),
			llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortNone)),
		)
		if _, present := body["thinking"]; present {
			t.Errorf("thinking must be omitted for effort none: %v", body["thinking"])
		}
		if body["temperature"] != 0.7 {
			t.Errorf("temperature lost: %v", body["temperature"])
		}
	})
}

func TestBuildParams_ToolsAndChoice(t *testing.T) {
	tool := llm.NewFuncTool("get_weather", "Weather for a city", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{"type": "string"},
		},
		"required":             []string{"location"},
		"additionalProperties": false,
	}, nil)

	body := marshalParams(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithTools(tool),
		llm.WithToolChoice(llm.ToolChoiceRequired),
	)

	tools, _ := body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %v", body["tools"])
	}
	def := tools[0].(map[string]any)
	if def["name"] != "get_weather" || def["description"] != "Weather for a city" {
		t.Errorf("unexpected tool definition: %v", def)
	}
	schema, _ := def["input_schema"].(map[string]any)
	if schema["type"] != "object" {
		t.Errorf("input_schema type must be object: %v", schema)
	}
	if _, ok := schema["properties"].(map[string]any)["location"]; !ok {
		t.Errorf("properties lost: %v", schema)
	}
	if required, _ := schema["required"].([]any); len(required) != 1 || required[0] != "location" {
		t.Errorf("required lost: %v", schema)
	}
	if schema["additionalProperties"] != false {
		t.Errorf("extra schema keywords must be forwarded: %v", schema)
	}

	choice, _ := body["tool_choice"].(map[string]any)
	if choice["type"] != "any" {
		t.Errorf("required must map to any, got %v", body["tool_choice"])
	}
}

func TestBuildParams_NoToolsNoChoice(t *testing.T) {
	body := marshalParams(t, llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if _, present := body["tool_choice"]; present {
		t.Errorf("tool_choice must be omitted without tools: %v", body["tool_choice"])
	}
}

func TestBuildParams_JSONSchemaOutput(t *testing.T) {
	schema := llm.NewResponseSchema("person", "A person", map[string]any{
		"type":       "object",
		"properties": map[string]any{"name": map[string]any{"type": "string"}},
	})
	body := marshalParams(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithJSONResponse(schema),
	)
	output, _ := body["output_config"].(map[string]any)
	format, _ := output["format"].(map[string]any)
	if format["type"] != "json_schema" {
		t.Fatalf("expected json_schema output format, got %v", body["output_config"])
	}
	if _, ok := format["schema"].(map[string]any)["properties"]; !ok {
		t.Errorf("schema lost: %v", format)
	}
}

func TestBuildParams_ExtraFields(t *testing.T) {
	body := marshalParams(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithExtraFields(map[string]any{"top_k": 5}),
	)
	if body["top_k"] != float64(5) {
		t.Errorf("extra field not forwarded: %v", body)
	}
}

func TestBuildParams_Attachments(t *testing.T) {
	image, err := llm.NewImageAttachment("image/png", "data:image/png;base64,aGVsbG8=", false)
	if err != nil {
		t.Fatal(err)
	}
	doc, err := llm.NewDocumentAttachment("text/plain", "aGVsbG8=", false)
	if err != nil {
		t.Fatal(err)
	}
	pdf, err := llm.NewDocumentAttachment("application/pdf", "https://example.com/a.pdf", true)
	if err != nil {
		t.Fatal(err)
	}

	body := marshalParams(t, llm.WithMessages(
		llm.NewMultimodalMessage(llm.RoleUser, "Look", image, doc, pdf),
	))
	blocks := blocksOf(t, messagesOf(t, body)[0])
	if len(blocks) != 4 {
		t.Fatalf("expected text + 3 attachment blocks, got %v", blocks)
	}
	imgSource, _ := blocks[1]["source"].(map[string]any)
	if blocks[1]["type"] != "image" || imgSource["type"] != "base64" || imgSource["data"] != "aGVsbG8=" || imgSource["media_type"] != "image/png" {
		t.Errorf("unexpected image block: %v", blocks[1])
	}
	docSource, _ := blocks[2]["source"].(map[string]any)
	if blocks[2]["type"] != "document" || docSource["type"] != "text" || docSource["data"] != "hello" {
		t.Errorf("unexpected text document block: %v", blocks[2])
	}
	pdfSource, _ := blocks[3]["source"].(map[string]any)
	if blocks[3]["type"] != "document" || pdfSource["type"] != "url" || pdfSource["url"] != "https://example.com/a.pdf" {
		t.Errorf("unexpected pdf block: %v", blocks[3])
	}
}

func TestBuildParams_AudioAttachmentRejected(t *testing.T) {
	audio, err := llm.NewAudioAttachment("audio/wav", "aGVsbG8=", false)
	if err != nil {
		t.Fatal(err)
	}
	_, err = buildParams(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMultimodalMessage(llm.RoleUser, "Listen", audio),
	)), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Fatal("expected audio attachments to be rejected")
	}
}

func TestNormalizeBaseURL(t *testing.T) {
	cases := map[string]string{
		"https://api.anthropic.com":     "https://api.anthropic.com/",
		"https://api.anthropic.com/":    "https://api.anthropic.com/",
		"https://api.anthropic.com/v1":  "https://api.anthropic.com/",
		"https://api.anthropic.com/v1/": "https://api.anthropic.com/",
		"https://proxy.local/anthropic": "https://proxy.local/anthropic/",
	}
	for input, expected := range cases {
		if got := normalizeBaseURL(input); got != expected {
			t.Errorf("normalizeBaseURL(%q) = %q, expected %q", input, got, expected)
		}
	}
}

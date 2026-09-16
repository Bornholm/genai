package anthropic

import (
	"encoding/json"
	"strings"
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
		// 80 % of 10000 is 8000, clamped so a quarter of max_tokens stays
		// available for the answer.
		thinking, _ := body["thinking"].(map[string]any)
		if thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(7500) {
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
		// Raised by the answer margin (a quarter of the budget), bounded
		// by the provider default.
		if body["max_tokens"] != float64(6000+1500) {
			t.Errorf("max_tokens must exceed the budget by the answer margin, got %v", body["max_tokens"])
		}
	})

	t.Run("explicit max_tokens is a ceiling: the budget is clamped under it", func(t *testing.T) {
		budget := 6000
		body := marshalParams(t,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithMaxCompletionTokens(3000),
			llm.WithReasoning(&llm.ReasoningOptions{MaxTokens: &budget}),
		)
		if body["max_tokens"] != float64(3000) {
			t.Errorf("the caller's max_tokens must be respected, got %v", body["max_tokens"])
		}
		// A quarter of 3000 is below the minimum margin of 1024, which wins.
		thinking, _ := body["thinking"].(map[string]any)
		if thinking["budget_tokens"] != float64(3000-1024) {
			t.Errorf("budget must leave an output margin under max_tokens, got %v", body["thinking"])
		}
	})

	t.Run("explicit max_tokens too small for the minimum budget is an error", func(t *testing.T) {
		_, err := buildParams(llm.NewChatCompletionOptions(
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithMaxCompletionTokens(1000),
			llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortLow)),
		), "claude-sonnet-5", DefaultMaxTokens)
		if err == nil {
			t.Fatal("expected an error when max_tokens cannot hold the minimum budget")
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

	t.Run("default max_tokens keeps an output margin for high efforts", func(t *testing.T) {
		for _, effort := range []llm.ReasoningEffort{llm.ReasoningEffortHigh, llm.ReasoningEffortXHigh} {
			body := marshalParams(t,
				llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
				llm.WithReasoning(llm.NewReasoningOptions(effort)),
			)
			thinking, _ := body["thinking"].(map[string]any)
			budget := thinking["budget_tokens"].(float64)
			maxTokens := body["max_tokens"].(float64)
			if maxTokens-budget < float64(minOutputMargin) {
				t.Errorf("%s: only %v tokens left for the answer (budget %v, max_tokens %v)", effort, maxTokens-budget, budget, maxTokens)
			}
		}
	})

	t.Run("enabled alone means medium effort", func(t *testing.T) {
		enabled := true
		body := marshalParams(t,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithReasoning(&llm.ReasoningOptions{Enabled: &enabled}),
		)
		thinking, _ := body["thinking"].(map[string]any)
		if thinking["budget_tokens"] != float64(DefaultMaxTokens/2) {
			t.Errorf("expected half of max_tokens, got %v", body["thinking"])
		}
	})

	t.Run("configured default max_tokens is the raise margin", func(t *testing.T) {
		budget := 20000
		params, err := buildParams(llm.NewChatCompletionOptions(
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithReasoning(&llm.ReasoningOptions{MaxTokens: &budget}),
		), "claude-sonnet-5", 16384)
		if err != nil {
			t.Fatal(err)
		}
		if params.MaxTokens != 20000+5000 {
			t.Errorf("max_tokens must be raised by a quarter of the budget, got %d", params.MaxTokens)
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

func TestBuildParams_JSONWithoutSchemaInstructsTheModel(t *testing.T) {
	body := marshalParams(t,
		llm.WithMessages(
			llm.NewMessage(llm.RoleSystem, "You are terse."),
			llm.NewMessage(llm.RoleUser, "Hi"),
		),
		llm.WithResponseFormat(llm.ResponseFormatJSON),
	)
	if _, present := body["output_config"]; present {
		t.Errorf("no output_config without a schema: %v", body["output_config"])
	}
	system, _ := body["system"].([]any)
	if len(system) != 2 || system[1].(map[string]any)["text"] != jsonModeInstruction {
		t.Errorf("schema-less JSON mode must append the JSON instruction after the caller's system prompt: %v", body["system"])
	}
}

func TestBuildParams_RequiredToolChoiceFallsBackToAutoWithThinking(t *testing.T) {
	tool := llm.NewFuncTool("t", "", map[string]any{"type": "object"}, nil)
	body := marshalParams(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithTools(tool),
		llm.WithToolChoice(llm.ToolChoiceRequired),
		llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortLow)),
	)
	if choice, _ := body["tool_choice"].(map[string]any); choice["type"] != "auto" {
		t.Errorf("tool_choice any is refused with thinking, expected auto, got %v", body["tool_choice"])
	}
}

func TestBuildParams_EffortClampsUnderALargeConfiguredDefault(t *testing.T) {
	params, err := buildParams(llm.NewChatCompletionOptions(
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortXHigh)),
	), "claude-sonnet-5", 64000)
	if err != nil {
		t.Fatal(err)
	}
	if params.MaxTokens != 64000 {
		t.Errorf("an effort must never raise max_tokens past the configured default, got %d", params.MaxTokens)
	}
	if budget := params.Thinking.OfEnabled.BudgetTokens; budget > 64000-16000 {
		t.Errorf("budget %d leaves less than a quarter of max_tokens for the answer", budget)
	}
}

func TestBuildParams_SeveralBreakpointsInOneMergedTurn(t *testing.T) {
	cc := &llm.CacheControl{Type: "ephemeral"}
	body := marshalParams(t, llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Hi"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_1", "a", `{}`), llm.NewToolCall("call_2", "b", `{}`)),
		&cachedToolMessage{BaseToolMessage: llm.NewToolMessage("call_1", llm.NewToolResult("one")), cc: cc},
		&cachedToolMessage{BaseToolMessage: llm.NewToolMessage("call_2", llm.NewToolResult("two")), cc: cc},
	))
	user := blocksOf(t, messagesOf(t, body)[2])
	if len(user) != 2 {
		t.Fatalf("expected both tool results in one turn, got %v", user)
	}
	for i, block := range user {
		if _, present := block["cache_control"]; !present {
			t.Errorf("breakpoint of tool result %d lost after merging: %v", i, block)
		}
	}
}

func TestBuildParams_TooManyBreakpointsIsRejected(t *testing.T) {
	cc := &llm.CacheControl{Type: "ephemeral"}
	_, err := buildParams(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMessageWithCacheControl(llm.RoleSystem, "s", cc),
		llm.NewMessageWithCacheControl(llm.RoleUser, "1", cc),
		llm.NewMessageWithCacheControl(llm.RoleAssistant, "2", cc),
		llm.NewMessageWithCacheControl(llm.RoleUser, "3", cc),
		llm.NewMessageWithCacheControl(llm.RoleAssistant, "4", cc),
		llm.NewMessage(llm.RoleUser, "5"),
	)), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Fatal("expected more than 4 breakpoints to be rejected locally")
	}
}

func TestBuildParams_MergedTurnsKeepTheAPIBlockOrder(t *testing.T) {
	call := llm.NewToolCall("call_1", "get_weather", `{}`)
	body := marshalParams(t, llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Hi"),
		// A plain assistant message followed by a reasoning tool call: the
		// merged assistant turn must still start with the thinking block.
		llm.NewMessage(llm.RoleAssistant, "Sure."),
		llm.NewReasoningToolCallsMessage("", []llm.ReasoningDetail{
			{Type: llm.ReasoningDetailTypeText, Text: "thinking", Signature: "sig"},
		}, call),
		// A user message slipped between the tool call and its result: the
		// merged user turn must still start with the tool_result.
		llm.NewMessage(llm.RoleUser, "Hurry up."),
		llm.NewToolMessage("call_1", llm.NewToolResult("Sunny")),
	))

	messages := messagesOf(t, body)
	if len(messages) != 3 {
		t.Fatalf("expected user/assistant/user, got %v", messages)
	}
	assistant := blocksOf(t, messages[1])
	if len(assistant) != 3 || assistant[0]["type"] != "thinking" || assistant[1]["type"] != "text" || assistant[2]["type"] != "tool_use" {
		t.Errorf("assistant turn must be [thinking, text, tool_use], got %v", assistant)
	}
	user := blocksOf(t, messages[2])
	if len(user) != 2 || user[0]["type"] != "tool_result" || user[1]["type"] != "text" {
		t.Errorf("user turn must be [tool_result, text], got %v", user)
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
		"":                              "https://api.anthropic.com/",
		"https://proxy.local/anthropic": "https://proxy.local/anthropic/",
	}
	for input, expected := range cases {
		if got := normalizeBaseURL(input); got != expected {
			t.Errorf("normalizeBaseURL(%q) = %q, expected %q", input, got, expected)
		}
	}
}

func TestBuildParams_TemperatureIsClampedToOne(t *testing.T) {
	body := marshalParams(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithTemperature(1.5),
	)
	if body["temperature"] != 1.0 {
		t.Errorf("temperature above 1 must be clamped, got %v", body["temperature"])
	}
}

func TestBuildParams_OnlySystemMessagesIsRejected(t *testing.T) {
	_, err := buildParams(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMessage(llm.RoleSystem, "You are terse."),
	)), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Fatal("expected a validation error for a conversation without any turn")
	}
}

func TestBuildParams_MalformedRequiredIsRejected(t *testing.T) {
	tool := llm.NewFuncTool("t", "", map[string]any{
		"type":       "object",
		"properties": map[string]any{"a": map[string]any{"type": "string"}},
		"required":   []any{"a", 42},
	}, nil)
	_, err := buildParams(llm.NewChatCompletionOptions(
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithTools(tool),
	), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Fatal("expected an error for a non-string entry in required")
	}
}

func TestBuildParams_CacheControlFollowsItsMessageIntoTheMergedTurn(t *testing.T) {
	call := llm.NewToolCall("call_1", "get_weather", `{}`)
	toolMsg := llm.NewToolMessage("call_1", llm.NewToolResult("Sunny"))
	body := marshalParams(t, llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Hi"),
		llm.NewToolCallsMessage(call),
		// The hint is on the tool message; the user message that follows
		// shares its turn but is not part of what the caller asked to
		// cache, so the breakpoint stays on the tool_result block.
		&cachedToolMessage{BaseToolMessage: toolMsg, cc: &llm.CacheControl{Type: "ephemeral"}},
		llm.NewMessage(llm.RoleUser, "Thanks."),
	))
	user := blocksOf(t, messagesOf(t, body)[2])
	if len(user) != 2 || user[0]["type"] != "tool_result" || user[1]["type"] != "text" {
		t.Fatalf("unexpected merged user turn: %v", user)
	}
	if cc, _ := user[0]["cache_control"].(map[string]any); cc["type"] != "ephemeral" {
		t.Errorf("breakpoint must stay on the block its message ended with: %v", user[0])
	}
	if _, present := user[1]["cache_control"]; present {
		t.Errorf("the following user text was not asked to be cached: %v", user[1])
	}
}

// cachedToolMessage is a tool message carrying a cache hint.
type cachedToolMessage struct {
	*llm.BaseToolMessage
	cc *llm.CacheControl
}

func (m *cachedToolMessage) CacheControl() *llm.CacheControl { return m.cc }

func TestBuildParams_ToolResultAttachments(t *testing.T) {
	image, err := llm.NewImageAttachment("image/png", "aGVsbG8=", false)
	if err != nil {
		t.Fatal(err)
	}
	body := marshalParams(t, llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Hi"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_1", "screenshot", `{}`)),
		llm.NewToolMessage("call_1", llm.NewToolResult("Here it is", image)),
	))
	result := blocksOf(t, messagesOf(t, body)[2])[0]
	content, _ := result["content"].([]any)
	if result["type"] != "tool_result" || len(content) != 2 {
		t.Fatalf("expected a tool_result with text + image, got %v", result)
	}
	if content[0].(map[string]any)["type"] != "text" || content[1].(map[string]any)["type"] != "image" {
		t.Errorf("unexpected tool_result content: %v", content)
	}

	audio, err := llm.NewAudioAttachment("audio/wav", "aGVsbG8=", false)
	if err != nil {
		t.Fatal(err)
	}
	_, err = buildParams(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Hi"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_1", "record", `{}`)),
		llm.NewToolMessage("call_1", llm.NewToolResult("", audio)),
	)), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Fatal("expected audio tool result attachments to be rejected")
	}
}

func TestAttachmentBlock_Rejections(t *testing.T) {
	svg, err := llm.NewImageAttachment("image/svg+xml", "aGVsbG8=", false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := attachmentBlock(svg); err == nil {
		t.Error("expected an unsupported image MIME type to be rejected")
	}

	if _, err := stripDataURL("data:text/plain,hello"); err == nil {
		t.Error("expected a non-base64 data URL to be rejected")
	}
	if payload, err := stripDataURL("data:image/png;base64,aGVsbG8="); err != nil || payload != "aGVsbG8=" {
		t.Errorf("base64 data URL not decoded: %q, %v", payload, err)
	}
}

func TestBuildParams_NonObjectToolSchemaIsRejected(t *testing.T) {
	tool := llm.NewFuncTool("t", "", map[string]any{"type": "array"}, nil)
	_, err := buildParams(llm.NewChatCompletionOptions(
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithTools(tool),
	), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Fatal("expected a non-object tool schema to be rejected")
	}
}

func TestBuildParams_CacheControlValidation(t *testing.T) {
	bad := "2h"
	_, err := buildParams(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMessageWithCacheControl(llm.RoleUser, "Hi", &llm.CacheControl{Type: "ephemeral", TTL: &bad}),
	)), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Error("expected an unsupported TTL to be rejected")
	}
	_, err = buildParams(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMessageWithCacheControl(llm.RoleUser, "Hi", &llm.CacheControl{Type: "persistent"}),
	)), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Error("expected an unsupported cache type to be rejected")
	}
}

func TestBuildParams_CacheHintOfAnEmptyMessageMovesToThePreviousTurn(t *testing.T) {
	body := marshalParams(t, llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Hi"),
		llm.NewMessage(llm.RoleAssistant, "Hello."),
		llm.NewMessageWithCacheControl(llm.RoleUser, "", &llm.CacheControl{Type: "ephemeral"}),
		llm.NewMessage(llm.RoleUser, "Next."),
	))
	messages := messagesOf(t, body)
	assistant := blocksOf(t, messages[1])
	if cc, _ := assistant[0]["cache_control"].(map[string]any); cc["type"] != "ephemeral" {
		t.Errorf("the hint of an empty message must land on the previous turn: %v", messages)
	}
}

func TestBuildParams_UnsignedReasoningOnlyTurnIsRejected(t *testing.T) {
	_, err := buildParams(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "A"),
		llm.NewAssistantReasoningMessage("", "thought", nil),
		llm.NewMessage(llm.RoleUser, "B"),
	)), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Fatal("expected an assistant turn made of unsigned reasoning only to be rejected rather than dropped")
	}
}

func TestBuildParams_BreakpointsAreCountedOncePlaced(t *testing.T) {
	cc := &llm.CacheControl{Type: "ephemeral"}
	// Five hints, but two resolve on the same block (empty message after an
	// annotated one) and one lands on a thinking-only turn: three markers.
	_, err := buildParams(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMessageWithCacheControl(llm.RoleSystem, "s", cc),
		llm.NewMessageWithCacheControl(llm.RoleUser, "1", cc),
		llm.NewMessageWithCacheControl(llm.RoleUser, "", cc),
		&cachedReasoningMessage{BaseAssistantReasoningMessage: llm.NewAssistantReasoningMessage("", "", []llm.ReasoningDetail{
			{Type: llm.ReasoningDetailTypeText, Text: "t", Signature: "sig"},
		}), cc: cc},
		llm.NewMessageWithCacheControl(llm.RoleUser, "3", cc),
	)), "claude-sonnet-5", DefaultMaxTokens)
	if err != nil {
		t.Fatalf("hints producing no marker must not count toward the limit: %v", err)
	}
}

// cachedReasoningMessage is an assistant reasoning message carrying a cache hint.
type cachedReasoningMessage struct {
	*llm.BaseAssistantReasoningMessage
	cc *llm.CacheControl
}

func (m *cachedReasoningMessage) CacheControl() *llm.CacheControl { return m.cc }

func TestBuildParams_ImageMimeTypeNormalisation(t *testing.T) {
	for _, mime := range []string{"image/jpg", "IMAGE/PNG"} {
		image, err := llm.NewImageAttachment(mime, "aGVsbG8=", false)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := attachmentBlock(image); err != nil {
			t.Errorf("%q must be accepted: %v", mime, err)
		}
	}
	// Parameters are refused upstream by llm's format validation, but the
	// normalisation copes with them for attachments built otherwise.
	if got := normalizeMimeType("image/webp; charset=binary"); got != "image/webp" {
		t.Errorf("parameters must be dropped, got %q", got)
	}
}

func TestBuildParams_ReasoningErrorNamesTheProviderDefault(t *testing.T) {
	_, err := buildParams(llm.NewChatCompletionOptions(
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortLow)),
	), "claude-sonnet-5", 1500)
	if err == nil || !strings.Contains(err.Error(), "MAX_TOKENS") {
		t.Errorf("the error must point at the provider default, got %v", err)
	}
}

func TestBuildParams_RaiseIsBounded(t *testing.T) {
	budget := 6000
	params, err := buildParams(llm.NewChatCompletionOptions(
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithReasoning(&llm.ReasoningOptions{MaxTokens: &budget}),
	), "claude-sonnet-5", DefaultMaxTokens)
	if err != nil {
		t.Fatal(err)
	}
	if params.MaxTokens != 6000+1500 {
		t.Errorf("expected budget + a quarter of it, got %d", params.MaxTokens)
	}
}

func TestBuildParams_EmptySystemMessageIsSkipped(t *testing.T) {
	cc := &llm.CacheControl{Type: "ephemeral"}
	body := marshalParams(t, llm.WithMessages(
		llm.NewMessage(llm.RoleSystem, "You are terse."),
		llm.NewMessageWithCacheControl(llm.RoleSystem, "", cc),
		llm.NewMessage(llm.RoleUser, "Hi"),
	))
	system, _ := body["system"].([]any)
	if len(system) != 1 {
		t.Fatalf("an empty system message must not reach the wire: %v", body["system"])
	}
	if c, _ := system[0].(map[string]any)["cache_control"].(map[string]any); c["type"] != "ephemeral" {
		t.Errorf("its cache hint must move to the previous system block: %v", system[0])
	}
}

func TestBuildParams_EmptyAssistantMessageIsRejected(t *testing.T) {
	_, err := buildParams(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "A"),
		llm.NewMessage(llm.RoleAssistant, ""),
		llm.NewMessage(llm.RoleUser, "B"),
	)), "claude-sonnet-5", DefaultMaxTokens)
	if err == nil {
		t.Fatal("expected an empty assistant turn to be rejected rather than folded away")
	}
}

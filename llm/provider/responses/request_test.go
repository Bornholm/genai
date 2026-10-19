package responses

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

// wire builds the request for opts and returns it as decoded JSON.
func wire(t *testing.T, funcs ...llm.ChatCompletionOptionFunc) map[string]any {
	t.Helper()
	req, err := buildRequest(llm.NewChatCompletionOptions(funcs...), "gpt-test", false)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return decoded
}

func TestBuildRequest_ConvertsAConversationWithToolsAndReasoning(t *testing.T) {
	details := []llm.ReasoningDetail{
		{ID: "rs_1", Type: llm.ReasoningDetailTypeSummary, Summary: "Look up the weather.", Format: reasoningFormat},
		{ID: "rs_1", Type: llm.ReasoningDetailTypeEncrypted, Data: "gAAA", Format: reasoningFormat},
		// Another provider's signed thinking cannot be replayed here.
		{Type: llm.ReasoningDetailTypeText, Text: "thinking", Signature: "sig"},
		// Nor can an item whose encrypted content was not kept.
		{ID: "rs_2", Type: llm.ReasoningDetailTypeSummary, Summary: "lost", Format: reasoningFormat},
	}

	body := wire(t,
		llm.WithMessages(
			llm.NewMessage(llm.RoleSystem, "Be brief."),
			llm.NewMessage(llm.RoleUser, "Weather in Paris?"),
			llm.NewReasoningToolCallsMessageWithContent("Checking.", "Look up the weather.", details,
				llm.NewToolCall("call_1", "weather", `{"city":"Paris"}`)),
			llm.NewToolMessage("call_1", llm.NewToolResult("18°C")),
		),
		llm.WithTools(llm.NewFuncTool("weather", "Current weather", llm.NewJSONSchema().RequiredProperty("city", "City", "string"), nil)),
		llm.WithToolChoice(llm.ToolChoiceRequired),
		llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortHigh)),
		llm.WithMaxCompletionTokens(500),
		llm.WithSessionID("session-1"),
		llm.WithExtraFields(map[string]any{"service_tier": "flex"}),
	)

	if body["model"] != "gpt-test" || body["store"] != false || body["max_output_tokens"] != float64(500) ||
		body["prompt_cache_key"] != "session-1" || body["tool_choice"] != "required" || body["service_tier"] != "flex" {
		t.Errorf("top-level fields = %v", body)
	}
	if include := body["include"].([]any); len(include) != 1 || include[0] != includeEncryptedReasoning {
		t.Errorf("include = %v", include)
	}
	if r := body["reasoning"].(map[string]any); r["effort"] != "high" || r["summary"] != "auto" {
		t.Errorf("reasoning = %v", r)
	}
	tools := body["tools"].([]any)
	if fn := tools[0].(map[string]any); fn["type"] != "function" || fn["name"] != "weather" || fn["strict"] != false || fn["parameters"] == nil {
		t.Errorf("tool = %v", fn)
	}

	input := body["input"].([]any)
	types := make([]string, 0, len(input))
	for _, item := range input {
		types = append(types, item.(map[string]any)["type"].(string))
	}
	want := []string{"message", "message", "reasoning", "message", "function_call", "function_call_output"}
	if len(types) != len(want) {
		t.Fatalf("input types = %v, want %v", types, want)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Fatalf("input types = %v, want %v", types, want)
		}
	}

	if system := input[0].(map[string]any); system["role"] != "system" || system["content"] != "Be brief." {
		t.Errorf("system = %v", system)
	}
	r := input[2].(map[string]any)
	summary := r["summary"].([]any)
	if r["id"] != "rs_1" || r["encrypted_content"] != "gAAA" || len(summary) != 1 ||
		summary[0].(map[string]any)["text"] != "Look up the weather." {
		t.Errorf("reasoning item = %v", r)
	}
	if text := input[3].(map[string]any); text["role"] != "assistant" || text["content"] != "Checking." {
		t.Errorf("assistant text = %v", text)
	}
	if call := input[4].(map[string]any); call["call_id"] != "call_1" || call["name"] != "weather" || call["arguments"] != `{"city":"Paris"}` {
		t.Errorf("function call = %v", call)
	}
	if output := input[5].(map[string]any); output["call_id"] != "call_1" || output["output"] != "18°C" {
		t.Errorf("function call output = %v", output)
	}
}

func TestBuildRequest_AsksForEncryptedReasoningOnlyWithReasoning(t *testing.T) {
	disabled := false
	for name, funcs := range map[string][]llm.ChatCompletionOptionFunc{
		"no reasoning":       nil,
		"disabled reasoning": {llm.WithReasoning(&llm.ReasoningOptions{Enabled: &disabled})},
	} {
		t.Run(name, func(t *testing.T) {
			body := wire(t, append(funcs, llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))...)
			if include, ok := body["include"]; ok {
				t.Errorf("include = %v, want none: models without reasoning refuse it", include)
			}
		})
	}
}

func TestBuildRequest_ConvertsAttachments(t *testing.T) {
	image, _ := llm.NewImageAttachment("image/png", "aGVsbG8=", false)
	remote, _ := llm.NewImageAttachment("image/jpeg", "https://example.com/cat.jpg", true)
	pdf, _ := llm.NewDocumentAttachment("application/pdf", "JVBERi0=", false)

	body := wire(t, llm.WithMessages(llm.NewMultimodalMessage(llm.RoleUser, "Describe.", image, remote, pdf)))

	content := body["input"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 4 {
		t.Fatalf("content = %v", content)
	}
	if part := content[1].(map[string]any); part["type"] != "input_image" || part["image_url"] != "data:image/png;base64,aGVsbG8=" {
		t.Errorf("inline image = %v", part)
	}
	if part := content[2].(map[string]any); part["image_url"] != "https://example.com/cat.jpg" {
		t.Errorf("remote image = %v", part)
	}
	if part := content[3].(map[string]any); part["type"] != "input_file" || part["file_data"] != "data:application/pdf;base64,JVBERi0=" || part["filename"] != "document.pdf" {
		t.Errorf("pdf = %v", part)
	}
}

func TestBuildRequest_RefusesAudio(t *testing.T) {
	audio, _ := llm.NewAudioAttachment("audio/wav", "UklGRg==", false)
	_, err := buildRequest(llm.NewChatCompletionOptions(
		llm.WithMessages(llm.NewMultimodalMessage(llm.RoleUser, "Listen.", audio)),
	), "gpt-test", false)
	if err == nil {
		t.Fatal("an audio attachment must be refused")
	}
}

func TestBuildRequest_JSONFormat(t *testing.T) {
	schema := llm.NewResponseSchema("answer", "The answer", map[string]any{"type": "object"}).WithStrict(false)
	body := wire(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "JSON please")),
		llm.WithJSONResponse(schema),
	)
	format := body["text"].(map[string]any)["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "answer" || format["strict"] != false || format["schema"] == nil {
		t.Errorf("format = %v", format)
	}

	body = wire(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "JSON please")),
		llm.WithResponseFormat(llm.ResponseFormatJSON),
	)
	if format := body["text"].(map[string]any)["format"].(map[string]any); format["type"] != "json_object" {
		t.Errorf("schema-less format = %v", format)
	}
}

func TestBuildReasoning(t *testing.T) {
	budget := func(n int) *int { return &n }
	disabled := false

	cases := map[string]struct {
		opts      *llm.ReasoningOptions
		maxTokens *int
		effort    string
		summary   string
	}{
		"effort":                   {opts: llm.NewReasoningOptions(llm.ReasoningEffortLow), effort: "low", summary: "auto"},
		"disabled":                 {opts: &llm.ReasoningOptions{Enabled: &disabled}, effort: "none"},
		"excluded":                 {opts: &llm.ReasoningOptions{Effort: ptr(llm.ReasoningEffortMedium), Exclude: true}, effort: "medium"},
		"budget share":             {opts: &llm.ReasoningOptions{MaxTokens: budget(800)}, maxTokens: budget(1000), effort: "high", summary: "auto"},
		"budget without a max":     {opts: &llm.ReasoningOptions{MaxTokens: budget(16000)}, effort: "high", summary: "auto"},
		"small budget without max": {opts: &llm.ReasoningOptions{MaxTokens: budget(1024)}, effort: "low", summary: "auto"},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			r := buildReasoning(c.opts, c.maxTokens)
			if r == nil || r.Effort != c.effort || r.Summary != c.summary {
				t.Errorf("reasoning = %+v, want effort %q summary %q", r, c.effort, c.summary)
			}
		})
	}

	if buildReasoning(nil, nil) != nil {
		t.Error("no reasoning options must send no reasoning")
	}
}

func TestToolCallArguments_ReplaysMalformedArgumentsAsAnEmptyObject(t *testing.T) {
	for _, parameters := range []any{nil, "", `{"a":1}{"b":2}`, []byte("{")} {
		got, err := toolCallArguments(parameters)
		if err != nil || got != "{}" {
			t.Errorf("toolCallArguments(%q) = %q, %v", parameters, got, err)
		}
	}
	if got, _ := toolCallArguments(`{"city":"Paris"}`); got != `{"city":"Paris"}` {
		t.Errorf("valid arguments must be kept, got %q", got)
	}
}

func ptr[T any](v T) *T { return &v }

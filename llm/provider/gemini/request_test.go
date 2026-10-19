package gemini

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

// wire builds the request for opts and returns it as decoded JSON.
func wire(t *testing.T, funcs ...llm.ChatCompletionOptionFunc) map[string]any {
	t.Helper()
	req, err := buildRequest(llm.NewChatCompletionOptions(funcs...))
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

func TestBuildRequest_ConvertsAConversationWithParallelToolCalls(t *testing.T) {
	details := []llm.ReasoningDetail{
		{ID: "call_1", Type: llm.ReasoningDetailTypeEncrypted, Data: "sig-1", Format: signatureFormat},
		// Another provider's signed thinking is not a thought signature.
		{Type: llm.ReasoningDetailTypeText, Text: "thinking", Signature: "sig"},
	}

	body := wire(t,
		llm.WithMessages(
			llm.NewMessage(llm.RoleSystem, "Be brief."),
			llm.NewMessage(llm.RoleUser, "Weather in Paris and Lyon?"),
			llm.NewReasoningToolCallsMessageWithContent("", "", details,
				llm.NewToolCall("call_1", "weather", `{"city":"Paris"}`),
				llm.NewToolCall("call_2", "weather", `{"city":"Lyon"}`)),
			llm.NewToolMessage("call_1", llm.NewToolResult("18°C")),
			llm.NewToolMessage("call_2", llm.NewToolResult(`{"celsius":21}`)),
			llm.NewMessage(llm.RoleSystem, "Answer in French."),
		),
		llm.WithTools(llm.NewFuncTool("weather", "Current weather", map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
		}, nil)),
		llm.WithToolChoice(llm.ToolChoiceRequired),
		llm.WithTemperature(0.2),
		llm.WithSeed(7),
		llm.WithExtraFields(map[string]any{"safetySettings": []any{}}),
	)

	system := body["systemInstruction"].(map[string]any)["parts"].([]any)
	if len(system) != 2 || system[1].(map[string]any)["text"] != "Answer in French." {
		t.Errorf("system instruction = %v", system)
	}

	contents := body["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents = %v, want user, model, user", contents)
	}
	model := contents[1].(map[string]any)
	calls := model["parts"].([]any)
	if model["role"] != "model" || len(calls) != 2 {
		t.Fatalf("model turn = %v", model)
	}
	first := calls[0].(map[string]any)
	if first["thoughtSignature"] != "sig-1" || first["functionCall"].(map[string]any)["args"].(map[string]any)["city"] != "Paris" {
		t.Errorf("first call = %v", first)
	}
	if second := calls[1].(map[string]any); second["thoughtSignature"] != nil {
		t.Errorf("only the first call carries a signature, got %v", second)
	}

	results := contents[2].(map[string]any)["parts"].([]any)
	if len(results) != 2 {
		t.Fatalf("both results must share one turn, got %v", results)
	}
	r1 := results[0].(map[string]any)["functionResponse"].(map[string]any)
	if r1["name"] != "weather" || r1["id"] != "call_1" || r1["response"].(map[string]any)["result"] != "18°C" {
		t.Errorf("first result = %v", r1)
	}
	if r2 := results[1].(map[string]any)["functionResponse"].(map[string]any); r2["response"].(map[string]any)["celsius"] != float64(21) {
		t.Errorf("a JSON object result must be sent as is, got %v", r2)
	}

	if mode := body["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)["mode"]; mode != "ANY" {
		t.Errorf("mode = %v", mode)
	}
	declaration := body["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	if declaration["name"] != "weather" || declaration["parametersJsonSchema"] == nil {
		t.Errorf("declaration = %v", declaration)
	}
	config := body["generationConfig"].(map[string]any)
	if config["temperature"] != 0.2 || config["seed"] != float64(7) {
		t.Errorf("generation config = %v", config)
	}
	if _, ok := body["safetySettings"]; !ok {
		t.Error("extra fields must be merged into the body")
	}
}

func TestBuildRequest_UserTextAfterToolResultsOpensATurn(t *testing.T) {
	body := wire(t, llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Ping?"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_1", "ping", "")),
		llm.NewToolMessage("call_1", llm.NewToolResult("pong")),
		llm.NewMessage(llm.RoleUser, "Now answer."),
	))

	contents := body["contents"].([]any)
	if len(contents) != 4 {
		t.Fatalf("contents = %v, want the tool results and the user text in two turns", contents)
	}
	if text := contents[3].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"]; text != "Now answer." {
		t.Errorf("last turn = %v", contents[3])
	}
}

func TestBuildRequest_PlaceholderSignatureForAForeignToolCall(t *testing.T) {
	body := wire(t, llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Ping?"),
		llm.NewToolCallsMessage(llm.NewToolCall("toolu_1", "ping", "")),
		llm.NewToolMessage("toolu_1", llm.NewToolResult("pong")),
	))

	call := body["contents"].([]any)[1].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if call["thoughtSignature"] != skipSignature {
		t.Errorf("call = %v", call)
	}
}

func TestBuildRequest_RefusesAnOrphanToolResult(t *testing.T) {
	_, err := buildRequest(llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMessage(llm.RoleUser, "Hi"),
		llm.NewToolMessage("call_unknown", llm.NewToolResult("?")),
	)))
	if err == nil {
		t.Fatal("a tool result answering no call must be refused")
	}
}

func TestBuildRequest_ConvertsAttachments(t *testing.T) {
	image, _ := llm.NewImageAttachment("image/jpg", "data:image/jpeg;base64,aGVsbG8=", false)
	audio, _ := llm.NewAudioAttachment("audio/wav", "UklGRg==", false)
	remote, _ := llm.NewDocumentAttachment("application/pdf", "https://example.com/doc.pdf", true)

	body := wire(t, llm.WithMessages(llm.NewMultimodalMessage(llm.RoleUser, "Look.", image, audio, remote)))

	parts := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if len(parts) != 4 {
		t.Fatalf("parts = %v", parts)
	}
	if inline := parts[1].(map[string]any)["inlineData"].(map[string]any); inline["mimeType"] != "image/jpeg" || inline["data"] != "aGVsbG8=" {
		t.Errorf("image = %v", inline)
	}
	if inline := parts[2].(map[string]any)["inlineData"].(map[string]any); inline["mimeType"] != "audio/wav" {
		t.Errorf("audio = %v", inline)
	}
	if file := parts[3].(map[string]any)["fileData"].(map[string]any); file["fileUri"] != "https://example.com/doc.pdf" {
		t.Errorf("document = %v", file)
	}
}

func TestBuildRequest_JSONAndThinking(t *testing.T) {
	schema := llm.NewResponseSchema("answer", "", map[string]any{"type": "object"})
	body := wire(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "JSON please")),
		llm.WithJSONResponse(schema),
		llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortHigh)),
		llm.WithMaxCompletionTokens(1000),
	)

	config := body["generationConfig"].(map[string]any)
	if config["responseMimeType"] != "application/json" || config["responseJsonSchema"] == nil {
		t.Errorf("config = %v", config)
	}
	thinking := config["thinkingConfig"].(map[string]any)
	if thinking["thinkingBudget"] != float64(800) || thinking["includeThoughts"] != true {
		t.Errorf("thinking = %v", thinking)
	}
}

func TestBuildThinking(t *testing.T) {
	disabled := false
	enabled := true
	budget := 2048

	cases := map[string]struct {
		opts      *llm.ReasoningOptions
		maxTokens int
		budget    int
		thoughts  bool
	}{
		"disabled":         {opts: &llm.ReasoningOptions{Enabled: &disabled}, budget: 0},
		"budget":           {opts: &llm.ReasoningOptions{MaxTokens: &budget}, budget: 2048, thoughts: true},
		"effort, no max":   {opts: llm.NewReasoningOptions(llm.ReasoningEffortLow), budget: 1024, thoughts: true},
		"enabled, dynamic": {opts: &llm.ReasoningOptions{Enabled: &enabled}, budget: dynamicBudget, thoughts: true},
		// Thoughts still come back, for their signatures; only the caller
		// does not see them.
		"excluded":        {opts: &llm.ReasoningOptions{MaxTokens: &budget, Exclude: true}, budget: 2048, thoughts: true},
		"effort, bounded": {opts: llm.NewReasoningOptions(llm.ReasoningEffortXHigh), maxTokens: 65536, budget: maxEffortBudget, thoughts: true},
		"effort, floored": {opts: llm.NewReasoningOptions(llm.ReasoningEffortLow), maxTokens: 300, budget: minEffortBudget, thoughts: true},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			var maxTokens *int
			if c.maxTokens > 0 {
				maxTokens = &c.maxTokens
			}
			config := buildThinking(c.opts, maxTokens)
			if config == nil || *config.ThinkingBudget != c.budget || config.IncludeThoughts != c.thoughts {
				t.Errorf("config = %+v, want budget %d thoughts %v", config, c.budget, c.thoughts)
			}
		})
	}
}

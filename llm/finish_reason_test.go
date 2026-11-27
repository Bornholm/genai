package llm

import "testing"

func TestParseOpenAIFinishReason(t *testing.T) {
	for value, want := range map[string]FinishReason{
		"stop":           FinishReasonStop,
		"length":         FinishReasonLength,
		"model_length":   FinishReasonLength,
		"tool_calls":     FinishReasonToolCalls,
		"function_call":  FinishReasonToolCalls,
		"content_filter": FinishReasonContentFilter,
		"error":          "",
		"":               "",
	} {
		if got := ParseOpenAIFinishReason(value); got != want {
			t.Errorf("ParseOpenAIFinishReason(%q) = %q, want %q", value, got, want)
		}
	}
}

func TestFinishReasonOf(t *testing.T) {
	if got := FinishReasonOf(NewCompleteStreamChunk(nil).WithFinishReason(FinishReasonLength)); got != FinishReasonLength {
		t.Errorf("FinishReasonOf(chunk) = %q, want %q", got, FinishReasonLength)
	}
	if got := FinishReasonOf(NewChatCompletionResponse(NewMessage(RoleAssistant, ""), nil)); got != "" {
		t.Errorf("FinishReasonOf(response without reason) = %q, want empty", got)
	}
	if got := FinishReasonOf(struct{}{}); got != "" {
		t.Errorf("FinishReasonOf(non-reporter) = %q, want empty", got)
	}
}

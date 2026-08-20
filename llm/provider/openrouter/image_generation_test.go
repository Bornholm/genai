package openrouter

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

// Images are billed per output, not per token: an estimate built from token
// counts lands nowhere near the real price. OpenRouter states the cost in
// the usage object, so dropping it would leave the caller guessing on the
// most expensive calls it makes.
func TestImageGenerationResponse_ReportsCost(t *testing.T) {
	var parsed imageGenerationResponse
	raw := `{"data":[{"b64_json":"AA==","media_type":"image/jpeg"}],
		"usage":{"prompt_tokens":3,"completion_tokens":1120,"total_tokens":1123,"cost":0.03360075}}`
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		t.Fatalf("decoding response: %v", err)
	}

	if parsed.Usage.Cost == nil {
		t.Fatal("cost not decoded")
	}

	usage := llm.NewImageGenerationUsageWithCost(
		parsed.Usage.PromptTokens, parsed.Usage.CompletionTokens, parsed.Usage.TotalTokens,
		*parsed.Usage.Cost, "USD",
	)

	amount, currency, ok := usage.Cost()
	if !ok {
		t.Fatal("cost not reported")
	}
	if amount != 0.03360075 {
		t.Errorf("cost = %v, want 0.03360075", amount)
	}
	if currency != "USD" {
		t.Errorf("currency = %q, want USD", currency)
	}
}

// A payload without cost claims none: a zero would read as a free image.
func TestImageGenerationResponse_NoCostReported(t *testing.T) {
	var parsed imageGenerationResponse
	raw := `{"data":[],"usage":{"prompt_tokens":3,"completion_tokens":0,"total_tokens":3}}`
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		t.Fatalf("decoding response: %v", err)
	}

	if parsed.Usage.Cost != nil {
		t.Errorf("cost = %v, want none", *parsed.Usage.Cost)
	}
}

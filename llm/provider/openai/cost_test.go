package openai

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
)

// usageFrom decodes a raw usage payload the way the SDK does, so the test
// exercises the same ExtraFields plumbing as a live response.
func usageFrom(t *testing.T, raw string) openai.CompletionUsage {
	t.Helper()

	var usage openai.CompletionUsage
	if err := json.Unmarshal([]byte(raw), &usage); err != nil {
		t.Fatalf("decoding usage: %v", err)
	}

	return usage
}

// Gateways such as OpenRouter report the price of a call in a non standard
// "cost" field. Ignoring it forces the caller to bill from an estimate,
// which is exactly what a reported cost is meant to replace.
func TestExtraCost(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
		want float64
		ok   bool
	}{
		{"reported cost", `{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.00042}`, 0.00042, true},
		{"free call", `{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0}`, 0, true},
		{"no cost reported", `{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}`, 0, false},
		{"malformed cost", `{"prompt_tokens":10,"cost":"cher"}`, 0, false},
		{"negative cost", `{"prompt_tokens":10,"cost":-1}`, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			usage := usageFrom(t, tc.raw)

			cost, ok := extraCost(usage.JSON.ExtraFields)
			if ok != tc.ok {
				t.Fatalf("cost reported = %v, want %v", ok, tc.ok)
			}
			if cost != tc.want {
				t.Errorf("cost = %v, want %v", cost, tc.want)
			}
		})
	}
}

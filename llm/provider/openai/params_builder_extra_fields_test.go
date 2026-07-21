package openai

import (
	"context"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
)

func TestConfigureExtraFields(t *testing.T) {
	t.Run("no extra fields leaves params untouched", func(t *testing.T) {
		params := &openai.ChatCompletionNewParams{}
		if err := ConfigureExtraFields(context.Background(), &llm.ChatCompletionOptions{}, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := params.GetExtraFields(); len(got) != 0 {
			t.Fatalf("expected no extra fields, got %v", got)
		}
	})

	t.Run("injects provided fields verbatim", func(t *testing.T) {
		params := &openai.ChatCompletionNewParams{}
		opts := &llm.ChatCompletionOptions{
			ExtraFields: map[string]any{"reasoning_split": true},
		}
		if err := ConfigureExtraFields(context.Background(), opts, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := params.GetExtraFields()
		if got["reasoning_split"] != true {
			t.Fatalf("expected reasoning_split=true, got %v", got)
		}
	})

	t.Run("merges with fields set by earlier configurators", func(t *testing.T) {
		params := &openai.ChatCompletionNewParams{}
		opts := &llm.ChatCompletionOptions{
			Reasoning:   llm.NewReasoningOptions(llm.ReasoningEffortHigh),
			ExtraFields: map[string]any{"reasoning_split": true},
		}
		// ConfigureReasoning sets reasoning_effort via WithExtraFields; because
		// WithExtraFields overwrites the whole map, ConfigureExtraFields must
		// merge rather than clobber it.
		if err := ConfigureReasoning(context.Background(), opts, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if err := ConfigureExtraFields(context.Background(), opts, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := params.GetExtraFields()
		if got["reasoning_effort"] != string(llm.ReasoningEffortHigh) {
			t.Fatalf("reasoning_effort was clobbered, got %v", got)
		}
		if got["reasoning_split"] != true {
			t.Fatalf("expected reasoning_split=true, got %v", got)
		}
	})
}

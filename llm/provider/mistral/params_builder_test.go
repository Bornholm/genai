package mistral

import (
	"context"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
)

func buildParams(t *testing.T, funcs ...llm.ChatCompletionOptionFunc) *openai.ChatCompletionNewParams {
	t.Helper()
	b := &paramsBuilder{model: "mistral-small-latest"}
	params, err := b.BuildParams(context.Background(), llm.NewChatCompletionOptions(funcs...))
	if err != nil {
		t.Fatalf("could not build params: %v", err)
	}
	return params
}

// TestBuildParams_CombinedProviderFields covers the three Mistral-specific
// fields being set together: params.WithExtraFields replaces the request's
// extra fields instead of adding to them, so setting them one by one used to
// keep only the last.
func TestBuildParams_CombinedProviderFields(t *testing.T) {
	params := buildParams(t,
		llm.WithMaxCompletionTokens(128),
		llm.WithSeed(7),
		llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortMedium)),
		llm.WithExtraFields(map[string]any{"top_p": 0.9}),
	)

	extra := params.GetExtraFields()
	for _, field := range []string{"max_tokens", "random_seed", "prompt_mode", "top_p"} {
		if _, present := extra[field]; !present {
			t.Errorf("%q was dropped: the extra fields overwrote each other", field)
		}
	}
}

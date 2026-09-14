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

// TestBuildParams_DropsOpenAIPlatformFields covers the 422 "Extra inputs are not
// permitted" Mistral answers when a client built against the OpenAI API sends
// platform fields — "store" above all — that the proxy forwards verbatim.
func TestBuildParams_DropsOpenAIPlatformFields(t *testing.T) {
	params := buildParams(t, llm.WithExtraFields(map[string]any{
		"store":               false,
		"metadata":            map[string]any{"run": "42"},
		"service_tier":        "auto",
		"logprobs":            true,
		"top_p":               0.9,
		"parallel_tool_calls": true,
		"safe_prompt":         true,
	}))

	extra := params.GetExtraFields()
	for _, field := range []string{"store", "metadata", "service_tier", "logprobs"} {
		if _, present := extra[field]; present {
			t.Errorf("%q was forwarded to Mistral, which answers 422 on it", field)
		}
	}
	for _, field := range []string{"top_p", "parallel_tool_calls", "safe_prompt"} {
		if _, present := extra[field]; !present {
			t.Errorf("%q was dropped, but Mistral supports it", field)
		}
	}
}

// TestBuildParams_KeepsProviderFields asserts the filter does not clobber the
// fields the Mistral configurators set themselves.
func TestBuildParams_KeepsProviderFields(t *testing.T) {
	params := buildParams(t,
		llm.WithMaxCompletionTokens(128),
		llm.WithSeed(7),
		llm.WithExtraFields(map[string]any{"store": false}),
	)

	extra := params.GetExtraFields()
	if _, present := extra["store"]; present {
		t.Error("store was forwarded to Mistral")
	}
	if _, present := extra["max_tokens"]; !present {
		t.Error("max_tokens set by configureMistralMaxTokens was lost")
	}
	if _, present := extra["random_seed"]; !present {
		t.Error("random_seed set by configureRandomSeed was lost")
	}
}

// TestBuildParams_OnlyUnsupportedExtraFields guards the path where filtering
// empties the caller's map: the provider's own fields must survive.
func TestBuildParams_OnlyUnsupportedExtraFields(t *testing.T) {
	params := buildParams(t,
		llm.WithSeed(3),
		llm.WithExtraFields(map[string]any{"store": false}),
	)

	if _, present := params.GetExtraFields()["random_seed"]; !present {
		t.Error("random_seed was lost when every caller-provided field was filtered out")
	}
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

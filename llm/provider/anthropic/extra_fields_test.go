package anthropic

import (
	"reflect"
	"testing"

	"github.com/bornholm/genai/llm"
)

func weatherTool() llm.Tool {
	return llm.NewFuncTool("weather", "Get the weather", map[string]any{"type": "object"}, nil)
}

func TestBuildParams_ExtraFields_TranslatesOpenAIFields(t *testing.T) {
	body := marshalParams(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithExtraFields(map[string]any{
			"stop":              []any{"END", "STOP"},
			"user":              "u1",
			"top_p":             0.9,
			"frequency_penalty": 0.5,
			"presence_penalty":  0.5,
			"logprobs":          true,
			"store":             true,
			"metadata":          map[string]any{"team": "a"},
			"service_tier":      "flex",
		}),
	)

	if got := body["stop_sequences"]; !reflect.DeepEqual(got, []any{"END", "STOP"}) {
		t.Errorf("stop_sequences = %v, want [END STOP]", got)
	}
	if got := body["metadata"]; !reflect.DeepEqual(got, map[string]any{"user_id": "u1"}) {
		t.Errorf("metadata = %v, want the user as user_id", got)
	}
	if got := body["top_p"]; got != 0.9 {
		t.Errorf("top_p = %v, want 0.9", got)
	}
	for _, key := range []string{"stop", "user", "frequency_penalty", "presence_penalty", "logprobs", "store", "service_tier"} {
		if _, ok := body[key]; ok {
			t.Errorf("%s reached the request body: %v", key, body)
		}
	}
}

func TestBuildParams_ExtraFields_StopAsAString(t *testing.T) {
	body := marshalParams(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithExtraFields(map[string]any{"stop": "END"}),
	)
	if got := body["stop_sequences"]; !reflect.DeepEqual(got, []any{"END"}) {
		t.Errorf("stop_sequences = %v, want [END]", got)
	}
}

func TestBuildParams_ExtraFields_KeepsAnthropicValues(t *testing.T) {
	body := marshalParams(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithExtraFields(map[string]any{
			"metadata":     map[string]any{"user_id": "u2"},
			"user":         "u1",
			"service_tier": "standard_only",
		}),
	)
	if got := body["metadata"]; !reflect.DeepEqual(got, map[string]any{"user_id": "u2"}) {
		t.Errorf("metadata = %v, want the metadata user_id to win over user", got)
	}
	if got := body["service_tier"]; got != "standard_only" {
		t.Errorf("service_tier = %v, want standard_only", got)
	}
}

func TestBuildParams_ExtraFields_TopPFollowsTheSamplingRules(t *testing.T) {
	t.Run("with a temperature", func(t *testing.T) {
		body := marshalParams(t,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithTemperature(0.5),
			llm.WithExtraFields(map[string]any{"top_p": 0.9}),
		)
		if _, ok := body["top_p"]; ok {
			t.Errorf("top_p = %v, want it dropped next to a temperature", body["top_p"])
		}
	})

	t.Run("with thinking", func(t *testing.T) {
		budget := 2048
		enabled := true
		body := marshalParams(t,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithMaxCompletionTokens(8192),
			llm.WithReasoning(&llm.ReasoningOptions{MaxTokens: &budget, Enabled: &enabled}),
			llm.WithExtraFields(map[string]any{"top_p": 0.9}),
		)
		if _, ok := body["top_p"]; ok {
			t.Errorf("top_p = %v, want it dropped under thinking", body["top_p"])
		}
	})
}

func TestBuildParams_ExtraFields_ToolChoiceNamingAFunction(t *testing.T) {
	body := marshalParams(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithTools(weatherTool()),
		llm.WithExtraFields(map[string]any{
			"tool_choice":         map[string]any{"type": "function", "function": map[string]any{"name": "weather"}},
			"parallel_tool_calls": false,
		}),
	)
	want := map[string]any{"type": "tool", "name": "weather", "disable_parallel_tool_use": true}
	if got := body["tool_choice"]; !reflect.DeepEqual(got, want) {
		t.Errorf("tool_choice = %v, want %v", got, want)
	}
	if _, ok := body["parallel_tool_calls"]; ok {
		t.Errorf("parallel_tool_calls reached the request body: %v", body)
	}
}

func TestBuildParams_ExtraFields_ParallelToolCalls(t *testing.T) {
	t.Run("on auto", func(t *testing.T) {
		body := marshalParams(t,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithTools(weatherTool()),
			llm.WithToolChoice(llm.ToolChoiceAuto),
			llm.WithExtraFields(map[string]any{"parallel_tool_calls": false}),
		)
		want := map[string]any{"type": "auto", "disable_parallel_tool_use": true}
		if got := body["tool_choice"]; !reflect.DeepEqual(got, want) {
			t.Errorf("tool_choice = %v, want %v", got, want)
		}
	})

	t.Run("without tools", func(t *testing.T) {
		body := marshalParams(t,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
			llm.WithExtraFields(map[string]any{"parallel_tool_calls": false}),
		)
		if _, ok := body["tool_choice"]; ok {
			t.Errorf("tool_choice = %v, want none without tools", body["tool_choice"])
		}
		if _, ok := body["parallel_tool_calls"]; ok {
			t.Errorf("parallel_tool_calls reached the request body: %v", body)
		}
	})
}

func TestBuildParams_ExtraFields_ToolChoiceUnderThinking(t *testing.T) {
	budget := 2048
	enabled := true
	body := marshalParams(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithMaxCompletionTokens(8192),
		llm.WithReasoning(&llm.ReasoningOptions{MaxTokens: &budget, Enabled: &enabled}),
		llm.WithTools(weatherTool()),
		llm.WithExtraFields(map[string]any{
			"tool_choice": map[string]any{"type": "function", "function": map[string]any{"name": "weather"}},
		}),
	)
	if got := body["tool_choice"]; !reflect.DeepEqual(got, map[string]any{"type": "auto"}) {
		t.Errorf("tool_choice = %v, want auto under thinking", got)
	}
}

package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"reflect"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

func weatherTool() llm.Tool {
	return llm.NewFuncTool("weather", "Get the weather", map[string]any{"type": "object"}, nil)
}

func TestBuildRequest_ExtraFields_TranslatesOpenAIFields(t *testing.T) {
	body := wire(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithTools(weatherTool()),
		llm.WithToolChoice(llm.ToolChoiceAuto),
		llm.WithTemperature(0.5),
		llm.WithExtraFields(map[string]any{
			"stop":                []any{"END", ""},
			"top_p":               0.9,
			"top_k":               40,
			"frequency_penalty":   0.5,
			"presence_penalty":    -0.5,
			"tool_choice":         map[string]any{"type": "function", "function": map[string]any{"name": "weather"}},
			"parallel_tool_calls": true,
			"user":                "u1",
			"logprobs":            true,
			"service_tier":        "flex",
			"reasoning":           map[string]any{"effort": "high"},
		}),
	)

	for _, key := range []string{"stop", "top_p", "top_k", "frequency_penalty", "presence_penalty", "tool_choice",
		"parallel_tool_calls", "user", "logprobs", "service_tier", "reasoning"} {
		if _, ok := body[key]; ok {
			t.Errorf("%s reached generateContent: %v", key, body)
		}
	}

	want := map[string]any{
		"temperature":      0.5,
		"topP":             0.9,
		"topK":             float64(40),
		"stopSequences":    []any{"END"},
		"frequencyPenalty": 0.5,
		"presencePenalty":  -0.5,
	}
	if got := body["generationConfig"]; !reflect.DeepEqual(got, want) {
		t.Errorf("generationConfig = %v, want %v", got, want)
	}

	wantTools := map[string]any{"functionCallingConfig": map[string]any{"mode": "ANY", "allowedFunctionNames": []any{"weather"}}}
	if got := body["toolConfig"]; !reflect.DeepEqual(got, wantTools) {
		t.Errorf("toolConfig = %v, want %v", got, wantTools)
	}
}

func TestBuildRequest_ExtraFields_LeavesOutZeroPenalties(t *testing.T) {
	body := wire(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithExtraFields(map[string]any{"frequency_penalty": 0, "presence_penalty": 0.0}),
	)
	if _, ok := body["generationConfig"]; ok {
		t.Errorf("generationConfig = %v, want none for zero penalties", body["generationConfig"])
	}
}

func TestBuildRequest_ExtraFields_ToolChoiceNamingAnUndeclaredTool(t *testing.T) {
	body := wire(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithTools(weatherTool()),
		llm.WithToolChoice(llm.ToolChoiceRequired),
		llm.WithExtraFields(map[string]any{
			"tool_choice": map[string]any{"type": "function", "function": map[string]any{"name": "forecast"}},
		}),
	)
	want := map[string]any{"functionCallingConfig": map[string]any{"mode": "ANY"}}
	if got := body["toolConfig"]; !reflect.DeepEqual(got, want) {
		t.Errorf("toolConfig = %v, want the typed %v", got, want)
	}
}

func TestBuildRequest_ExtraFields_KeepsGeminiValues(t *testing.T) {
	body := wire(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithTemperature(0.5),
		llm.WithExtraFields(map[string]any{
			"top_p":            0.9,
			"safetySettings":   []any{map[string]any{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}},
			"cachedContent":    "cachedContents/abc",
			"generationConfig": map[string]any{"candidateCount": 1, "temperature": 0.2},
		}),
	)

	if body["cachedContent"] != "cachedContents/abc" || body["safetySettings"] == nil {
		t.Errorf("body = %v, want cachedContent and safetySettings forwarded", body)
	}
	want := map[string]any{"temperature": 0.2, "topP": 0.9, "candidateCount": float64(1)}
	if got := body["generationConfig"]; !reflect.DeepEqual(got, want) {
		t.Errorf("generationConfig = %v, want the forwarded one merged over %v", got, want)
	}
}

func TestBuildRequest_ExtraFields_DropsMalformedValues(t *testing.T) {
	baseline := wire(t, llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")), llm.WithTools(weatherTool()))

	for key, value := range map[string]any{
		"stop":                []any{"END", 1},
		"top_p":               "high",
		"top_k":               40.5,
		"frequency_penalty":   "none",
		"tool_choice":         "required",
		"parallel_tool_calls": false,
		"generation_config":   map[string]any{"topP": 0.5},
		"generationConfig":    "hot",
	} {
		t.Run(key, func(t *testing.T) {
			body := wire(t,
				llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
				llm.WithTools(weatherTool()),
				llm.WithExtraFields(map[string]any{key: value}),
			)
			if !reflect.DeepEqual(body, baseline) {
				t.Errorf("body = %v, want %v", body, baseline)
			}
		})
	}
}

func TestBuildRequest_ExtraFields_TopKOutOfRange(t *testing.T) {
	body := wire(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithExtraFields(map[string]any{"top_k": 1e30}),
	)
	if _, ok := body["generationConfig"]; ok {
		t.Errorf("generationConfig = %v, want top_k dropped", body["generationConfig"])
	}
}

func TestBuildRequest_ExtraFields_LogsEveryDrop(t *testing.T) {
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(previous)

	_, err := buildRequest(context.Background(), llm.NewChatCompletionOptions(
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithExtraFields(map[string]any{
			"user":                "u1",
			"parallel_tool_calls": false,
			"tool_choice":         map[string]any{"type": "function", "function": map[string]any{"name": "weather"}},
			"repetition_penalty":  1.1,
			"top_p":               0.9,
		}),
	))
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	var dropped []string
	for _, line := range strings.Split(strings.TrimSpace(logs.String()), "\n") {
		var entry struct {
			Msg    string   `json:"msg"`
			Fields []string `json:"fields"`
		}
		if json.Unmarshal([]byte(line), &entry) == nil && strings.HasPrefix(entry.Msg, "dropped request fields") {
			dropped = entry.Fields
		}
	}
	want := []string{"parallel_tool_calls", "repetition_penalty", "tool_choice", "user"}
	if !reflect.DeepEqual(dropped, want) {
		t.Errorf("dropped = %v, want %v", dropped, want)
	}
}

func TestBuildRequest_ExtraFields_ForwardedGenerationConfigInSnakeCase(t *testing.T) {
	body := wire(t,
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
		llm.WithTemperature(0.5),
		llm.WithExtraFields(map[string]any{
			"top_p":            0.9,
			"generationConfig": map[string]any{"top_p": 0.5, "stop_sequences": []any{"END"}, "candidateCount": 1},
		}),
	)
	want := map[string]any{"temperature": 0.5, "topP": 0.5, "stopSequences": []any{"END"}, "candidateCount": float64(1)}
	if got := body["generationConfig"]; !reflect.DeepEqual(got, want) {
		t.Errorf("generationConfig = %v, want %v", got, want)
	}
}

func TestBuildRequest_SendsAZeroGenerationConfigValue(t *testing.T) {
	disabled := false
	for name, opt := range map[string]llm.ChatCompletionOptionFunc{
		"temperature": llm.WithTemperature(0),
		"reasoning":   llm.WithReasoning(&llm.ReasoningOptions{Enabled: &disabled}),
	} {
		t.Run(name, func(t *testing.T) {
			body := wire(t, llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")), opt)
			if _, ok := body["generationConfig"]; !ok {
				t.Errorf("body = %v, want a generationConfig", body)
			}
		})
	}
}

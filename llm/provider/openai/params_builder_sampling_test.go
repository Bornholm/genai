package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
)

func marshalParams(t *testing.T, params *openai.ChatCompletionNewParams) map[string]any {
	t.Helper()
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	return out
}

func TestConfigureTemperature(t *testing.T) {
	t.Run("omitted when the caller did not set one", func(t *testing.T) {
		params := &openai.ChatCompletionNewParams{}
		opts := llm.NewChatCompletionOptions()
		if err := ConfigureTemperature(context.Background(), opts, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, ok := marshalParams(t, params)["temperature"]; ok {
			t.Fatalf("temperature must not be sent when unset")
		}
	})

	t.Run("forwarded when set, zero included", func(t *testing.T) {
		params := &openai.ChatCompletionNewParams{}
		opts := llm.NewChatCompletionOptions(llm.WithTemperature(0))
		if err := ConfigureTemperature(context.Background(), opts, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got, ok := marshalParams(t, params)["temperature"]
		if !ok || got != float64(0) {
			t.Fatalf("temperature = %v (present=%v), want 0", got, ok)
		}
	})
}

func TestConfigureResponseFormat(t *testing.T) {
	t.Run("json without schema falls back to json_object", func(t *testing.T) {
		params := &openai.ChatCompletionNewParams{}
		opts := llm.NewChatCompletionOptions(llm.WithResponseFormat(llm.ResponseFormatJSON))
		if err := ConfigureResponseFormat(context.Background(), opts, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		rf, _ := marshalParams(t, params)["response_format"].(map[string]any)
		if rf["type"] != "json_object" {
			t.Fatalf("response_format = %v, want json_object", rf)
		}
	})

	t.Run("schema is sent as json_schema with the requested strict mode", func(t *testing.T) {
		schema := llm.NewResponseSchema("answer", "", map[string]any{"type": "object"}).WithStrict(false)
		params := &openai.ChatCompletionNewParams{}
		opts := llm.NewChatCompletionOptions(llm.WithJSONResponse(schema))
		if err := ConfigureResponseFormat(context.Background(), opts, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		rf, _ := marshalParams(t, params)["response_format"].(map[string]any)
		if rf["type"] != "json_schema" {
			t.Fatalf("response_format type = %v, want json_schema", rf["type"])
		}
		js, _ := rf["json_schema"].(map[string]any)
		if js["name"] != "answer" {
			t.Fatalf("json_schema.name = %v, want answer", js["name"])
		}
		if js["strict"] != false {
			t.Fatalf("json_schema.strict = %v, want false", js["strict"])
		}
		if _, ok := js["description"]; ok {
			t.Fatalf("empty description must be omitted, got %v", js)
		}
	})

	t.Run("schemas without a strict preference stay strict", func(t *testing.T) {
		params := &openai.ChatCompletionNewParams{}
		opts := llm.NewChatCompletionOptions(llm.WithJSONResponse(plainSchema{}))
		if err := ConfigureResponseFormat(context.Background(), opts, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		raw, _ := json.Marshal(params)
		if !strings.Contains(string(raw), `"strict":true`) {
			t.Fatalf("expected strict:true, got %s", raw)
		}
	})
}

// plainSchema implements llm.ResponseSchema without StrictResponseSchema.
type plainSchema struct{}

func (plainSchema) Name() string        { return "plain" }
func (plainSchema) Description() string { return "" }
func (plainSchema) Schema() any         { return map[string]any{"type": "object"} }

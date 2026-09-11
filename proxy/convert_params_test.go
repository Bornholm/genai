package proxy

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

func TestParseChatCompletionRequest_TemperatureOptional(t *testing.T) {
	body := json.RawMessage(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	_, _, opts, err := ParseChatCompletionRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if compiled := llm.NewChatCompletionOptions(opts...); compiled.Temperature != nil {
		t.Fatalf("temperature = %v, want nil when the client omits it", *compiled.Temperature)
	}

	body = json.RawMessage(`{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0}`)
	_, _, opts, err = ParseChatCompletionRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	compiled := llm.NewChatCompletionOptions(opts...)
	if compiled.Temperature == nil || *compiled.Temperature != 0 {
		t.Fatalf("temperature = %v, want 0", compiled.Temperature)
	}
	if _, leaked := compiled.ExtraFields["temperature"]; leaked {
		t.Fatalf("temperature must not be duplicated in extra fields")
	}
}

func TestParseChatCompletionRequest_MaxCompletionTokens(t *testing.T) {
	body := json.RawMessage(`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"max_completion_tokens":20}`)
	_, _, opts, err := ParseChatCompletionRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	compiled := llm.NewChatCompletionOptions(opts...)
	if compiled.MaxCompletionTokens == nil || *compiled.MaxCompletionTokens != 20 {
		t.Fatalf("max completion tokens = %v, want 20 (max_completion_tokens wins)", compiled.MaxCompletionTokens)
	}
	if _, leaked := compiled.ExtraFields["max_completion_tokens"]; leaked {
		t.Fatalf("max_completion_tokens must not be duplicated in extra fields")
	}
}

func TestParseChatCompletionRequest_ResponseFormat(t *testing.T) {
	t.Run("json_object", func(t *testing.T) {
		body := json.RawMessage(`{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}`)
		_, _, opts, err := ParseChatCompletionRequest(body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		compiled := llm.NewChatCompletionOptions(opts...)
		if compiled.ResponseFormat != llm.ResponseFormatJSON || compiled.ResponseSchema != nil {
			t.Fatalf("got format=%v schema=%v, want JSON without schema", compiled.ResponseFormat, compiled.ResponseSchema)
		}
	})

	t.Run("json_schema", func(t *testing.T) {
		body := json.RawMessage(`{"model":"m","messages":[{"role":"user","content":"hi"}],
			"response_format":{"type":"json_schema","json_schema":{"name":"answer","description":"An answer","strict":true,
				"schema":{"type":"object","properties":{"ok":{"type":"boolean"}}}}}}`)
		_, _, opts, err := ParseChatCompletionRequest(body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		compiled := llm.NewChatCompletionOptions(opts...)
		if compiled.ResponseFormat != llm.ResponseFormatJSON || compiled.ResponseSchema == nil {
			t.Fatalf("got format=%v schema=%v, want JSON with schema", compiled.ResponseFormat, compiled.ResponseSchema)
		}
		if compiled.ResponseSchema.Name() != "answer" || compiled.ResponseSchema.Description() != "An answer" {
			t.Fatalf("schema identity = %q / %q", compiled.ResponseSchema.Name(), compiled.ResponseSchema.Description())
		}
		if !llm.IsStrictResponseSchema(compiled.ResponseSchema) {
			t.Fatalf("strict:true was not honoured")
		}
		schema, _ := compiled.ResponseSchema.Schema().(map[string]any)
		if schema["type"] != "object" {
			t.Fatalf("schema = %v", compiled.ResponseSchema.Schema())
		}
	})

	t.Run("json_schema defaults to non strict", func(t *testing.T) {
		body := json.RawMessage(`{"model":"m","messages":[{"role":"user","content":"hi"}],
			"response_format":{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"}}}}`)
		_, _, opts, err := ParseChatCompletionRequest(body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		compiled := llm.NewChatCompletionOptions(opts...)
		if llm.IsStrictResponseSchema(compiled.ResponseSchema) {
			t.Fatalf("strict must default to false as it does at OpenAI")
		}
	})

	t.Run("json_schema without block is rejected", func(t *testing.T) {
		body := json.RawMessage(`{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema"}}`)
		if _, _, _, err := ParseChatCompletionRequest(body); err == nil {
			t.Fatalf("expected an error")
		}
	})
}

func TestParseMessagesRequest_SamplingPassthrough(t *testing.T) {
	body := json.RawMessage(`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],
		"top_p":0.9,"top_k":40,"stop_sequences":["END"]}`)
	_, _, opts, err := ParseMessagesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	compiled := llm.NewChatCompletionOptions(opts...)
	if compiled.Temperature != nil {
		t.Fatalf("temperature = %v, want nil", *compiled.Temperature)
	}
	if got := compiled.ExtraFields["top_p"]; got != 0.9 {
		t.Fatalf("top_p = %v, want 0.9", got)
	}
	stop, _ := compiled.ExtraFields["stop"].([]string)
	if len(stop) != 1 || stop[0] != "END" {
		t.Fatalf("stop = %v, want [END]", compiled.ExtraFields["stop"])
	}
	if _, ok := compiled.ExtraFields["top_k"]; ok {
		t.Fatalf("top_k has no OpenAI equivalent and must not be forwarded")
	}
}

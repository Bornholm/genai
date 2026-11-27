package gemini

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bornholm/genai/llm"
)

func TestChatCompletion_ReportsTheFinishReason(t *testing.T) {
	for reason, want := range map[string]llm.FinishReason{
		"STOP":       llm.FinishReasonStop,
		"MAX_TOKENS": llm.FinishReasonLength,
		"SAFETY":     llm.FinishReasonContentFilter,
	} {
		t.Run(reason, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]},"finishReason":"`+reason+`"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}`)
			}))
			defer server.Close()

			res, err := NewChatCompletionClient(nil, server.URL, "key", "gemini-test").
				ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
			if err != nil {
				t.Fatalf("chat completion: %v", err)
			}
			if got := llm.FinishReasonOf(res); got != want {
				t.Errorf("finish reason = %q, want %q", got, want)
			}
		})
	}
}

func TestChatCompletionStream_ReportsTheFinishReason(t *testing.T) {
	for reason, want := range map[string]llm.FinishReason{
		"STOP":       llm.FinishReasonStop,
		"MAX_TOKENS": llm.FinishReasonLength,
		"SAFETY":     llm.FinishReasonContentFilter,
	} {
		t.Run(reason, func(t *testing.T) {
			server := streamServer(t,
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}`,
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"finishReason":"`+reason+`"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}`,
			)
			defer server.Close()

			got := collect(t, server)
			if !got.terminal.IsComplete() {
				t.Fatalf("terminal = %v", got.terminal)
			}
			if reason := llm.FinishReasonOf(got.terminal); reason != want {
				t.Errorf("finish reason = %q, want %q", reason, want)
			}
		})
	}
}

func TestParseFinishReason(t *testing.T) {
	for reason, want := range map[string]llm.FinishReason{
		"STOP":                    llm.FinishReasonStop,
		"MAX_TOKENS":              llm.FinishReasonLength,
		"RECITATION":              llm.FinishReasonContentFilter,
		"PROHIBITED_CONTENT":      llm.FinishReasonContentFilter,
		"MALFORMED_FUNCTION_CALL": "",
		"":                        "",
	} {
		if got := parseFinishReason(reason); got != want {
			t.Errorf("parseFinishReason(%q) = %q, want %q", reason, got, want)
		}
	}
}

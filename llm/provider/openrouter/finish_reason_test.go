package openrouter

import (
	"context"
	"net/http"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/revrost/go-openrouter"
)

// TestChatCompletionStream_ReportsTheFinishReason asserts that the finish
// reason survives the usage-only chunk OpenRouter sends after it.
func TestChatCompletionStream_ReportsTheFinishReason(t *testing.T) {
	for value, want := range map[string]llm.FinishReason{
		"stop":           llm.FinishReasonStop,
		"length":         llm.FinishReasonLength,
		"tool_calls":     llm.FinishReasonToolCalls,
		"content_filter": llm.FinishReasonContentFilter,
	} {
		t.Run(value, func(t *testing.T) {
			client := newTestClient(t, sse(
				helloChunk,
				`{"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"`+value+`"}]}`,
				usageChunk,
				"[DONE]",
			))

			_, last := collect(t, client)
			if last == nil || !last.IsComplete() {
				t.Fatalf("last chunk = %v, want the completion chunk", last)
			}
			if got := llm.FinishReasonOf(last); got != want {
				t.Errorf("finish reason = %q, want %q", got, want)
			}
		})
	}
}

func TestParseFinishReason(t *testing.T) {
	cases := []struct {
		normalized, native string
		want               llm.FinishReason
	}{
		{"length", "MAX_TOKENS", llm.FinishReasonLength},
		// A choice carrying only the upstream provider's value.
		{"", "MAX_TOKENS", llm.FinishReasonLength},
		{"", "end_turn", llm.FinishReasonStop},
		{"", "tool_use", llm.FinishReasonToolCalls},
		// The normalized value wins over the native one.
		{"stop", "max_tokens", llm.FinishReasonStop},
		{"", "SAFETY", llm.FinishReasonContentFilter},
		{"", "MALFORMED_FUNCTION_CALL", ""},
		{"", "", ""},
	}
	for _, c := range cases {
		if got := parseFinishReason(openrouter.FinishReason(c.normalized), c.native); got != c.want {
			t.Errorf("parseFinishReason(%q, %q) = %q, want %q", c.normalized, c.native, got, c.want)
		}
	}
}

// TestChatCompletionStream_ReportsANativeOnlyFinishReason asserts that a
// choice carrying only native_finish_reason still reports its reason, the
// stream being taken as finished on that value already.
func TestChatCompletionStream_ReportsANativeOnlyFinishReason(t *testing.T) {
	client := newTestClient(t, sse(
		helloChunk,
		`{"choices":[{"index":0,"delta":{"content":"!"},"native_finish_reason":"MAX_TOKENS"}]}`,
		usageChunk,
		"[DONE]",
	))

	_, last := collect(t, client)
	if last == nil || !last.IsComplete() {
		t.Fatalf("last chunk = %v, want the completion chunk", last)
	}
	if got := llm.FinishReasonOf(last); got != llm.FinishReasonLength {
		t.Errorf("finish reason = %q, want %q", got, llm.FinishReasonLength)
	}
}

// TestChatCompletionStream_KeepsTheReasonOfTheFirstChoice asserts that the
// reason of a choice the consumer never receives is not reported.
func TestChatCompletionStream_KeepsTheReasonOfTheFirstChoice(t *testing.T) {
	client := newTestClient(t, sse(
		helloChunk,
		`{"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"},{"index":1,"delta":{"content":"?"},"finish_reason":"length"}]}`,
		usageChunk,
		"[DONE]",
	))

	_, last := collect(t, client)
	if got := llm.FinishReasonOf(last); got != llm.FinishReasonStop {
		t.Errorf("finish reason = %q, want %q", got, llm.FinishReasonStop)
	}
}

func TestChatCompletion_ReportsTheFinishReason(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"gen-1","choices":[{"index":0,"message":{"role":"assistant","content":"Hel"},"finish_reason":"length","native_finish_reason":"MAX_TOKENS"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`))
	})

	res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "hi")))
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if got := llm.FinishReasonOf(res); got != llm.FinishReasonLength {
		t.Errorf("finish reason = %q, want %q", got, llm.FinishReasonLength)
	}
}

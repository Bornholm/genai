package responses

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bornholm/genai/llm"
)

func TestChatCompletion_ReportsTheFinishReason(t *testing.T) {
	cases := map[string]struct {
		status, details string
		want            llm.FinishReason
	}{
		"completed":         {"completed", `null`, llm.FinishReasonStop},
		"max output tokens": {"incomplete", `{"reason":"max_output_tokens"}`, llm.FinishReasonLength},
		"content filter":    {"incomplete", `{"reason":"content_filter"}`, llm.FinishReasonContentFilter},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"resp_1","status":"`+c.status+`","incomplete_details":`+c.details+`,"output":[
					{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"Hel"}]}
				],"usage":{"input_tokens":3,"output_tokens":1,"total_tokens":4}}`)
			}))
			defer server.Close()

			res, err := NewChatCompletionClient(nil, server.URL, "sk-test", "gpt-test").
				ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
			if err != nil {
				t.Fatalf("chat completion: %v", err)
			}
			if got := llm.FinishReasonOf(res); got != c.want {
				t.Errorf("finish reason = %q, want %q", got, c.want)
			}
		})
	}
}

func TestChatCompletionStream_ReportsTheFinishReason(t *testing.T) {
	cases := map[string]struct {
		terminal string
		want     llm.FinishReason
	}{
		"completed": {
			`{"type":"response.completed","response":{"id":"resp_1","status":"completed"}}`,
			llm.FinishReasonStop,
		},
		"incomplete": {
			`{"type":"response.incomplete","response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`,
			llm.FinishReasonLength,
		},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			server := streamServer(t,
				`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}`,
				`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hel"}`,
				c.terminal,
			)
			defer server.Close()

			got := collect(t, server)
			if !got.terminal.IsComplete() {
				t.Fatalf("terminal = %v", got.terminal)
			}
			if reason := llm.FinishReasonOf(got.terminal); reason != c.want {
				t.Errorf("finish reason = %q, want %q", reason, c.want)
			}
		})
	}
}

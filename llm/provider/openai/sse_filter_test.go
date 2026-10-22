package openai_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/mistral"
	"github.com/bornholm/genai/llm/provider/openai"
)

func contentChunk(text string) string {
	return `data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"` + text + `"}}]}`
}

func TestChatCompletionStream_IgnoresEventsWithoutData(t *testing.T) {
	cases := []struct {
		name string
		// segments are written and flushed one after the other.
		segments []string
	}{
		{
			name:     "comments",
			segments: []string{": PROCESSING\n\n" + contentChunk("Hel") + "\n\n: PROCESSING\n\n" + contentChunk("lo") + "\n\ndata: [DONE]\n\n"},
		},
		{
			name:     "crlf comments",
			segments: []string{": PROCESSING\r\n\r\n" + contentChunk("Hel") + "\r\n\r\n" + contentChunk("lo") + "\r\n\r\ndata: [DONE]\r\n\r\n"},
		},
		{
			name:     "comment inside an event",
			segments: []string{contentChunk("Hel") + "\n: comment\n\n\n" + contentChunk("lo") + "\n\ndata: [DONE]\n\n"},
		},
		{
			name:     "fields without data",
			segments: []string{"event: ping\n\n" + "id: 42\n\n" + "retry: 1000\n\n" + "data:\n\n" + contentChunk("Hel") + "\n\nevent: ping\n\n" + contentChunk("lo") + "\n\ndata: [DONE]\n\n"},
		},
		{
			name: "split across writes",
			segments: []string{
				": PROC", "ESSING\n", "\n" + contentChunk("Hel")[:20], contentChunk("Hel")[20:] + "\n",
				"\n: PROCESSING\n\n", contentChunk("lo") + "\n\n", "data: [DONE]\n\n",
			},
		},
		{
			name:     "no final newline",
			segments: []string{": PROCESSING\n\n" + contentChunk("Hel") + "\n\n" + contentChunk("lo") + "\n\ndata: [DONE]"},
		},
	}

	providers := []struct {
		name   provider.Name
		create func(baseURL string) provider.OptionFunc
	}{
		{openai.Name, func(baseURL string) provider.OptionFunc {
			return provider.WithChatCompletion(openai.Name, openai.Options{
				CommonOptions: provider.CommonOptions{BaseURL: baseURL, APIKey: "sk-test", Model: "deepseek"},
			})
		}},
		{mistral.Name, func(baseURL string) provider.OptionFunc {
			return provider.WithChatCompletion(mistral.Name, mistral.Options{
				CommonOptions: provider.CommonOptions{BaseURL: baseURL, APIKey: "sk-test", Model: "mistral-small"},
			})
		}},
	}

	for _, p := range providers {
		for _, c := range cases {
			t.Run(string(p.name)+"/"+c.name, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
					for _, segment := range c.segments {
						_, _ = w.Write([]byte(segment))
						w.(http.Flusher).Flush()
					}
				}))
				defer server.Close()

				client, err := provider.Create(context.Background(), p.create(server.URL))
				if err != nil {
					t.Fatalf("create: %v", err)
				}

				chunks, err := client.ChatCompletionStream(context.Background(),
					llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")),
				)
				if err != nil {
					t.Fatalf("stream: %v", err)
				}

				var content strings.Builder
				complete := false
				for chunk := range chunks {
					if err := chunk.Error(); err != nil {
						t.Fatalf("stream chunk error: %v", err)
					}
					if d := chunk.Delta(); d != nil {
						content.WriteString(d.Content())
					}
					complete = complete || chunk.IsComplete()
				}

				if content.String() != "Hello" {
					t.Errorf("content = %q, want %q", content.String(), "Hello")
				}
				if !complete {
					t.Error("the stream must end with a completion chunk")
				}
			})
		}
	}
}

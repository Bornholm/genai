package openai_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/mistral"
	"github.com/bornholm/genai/llm/provider/openai"
)

// openAICompatibleProviders are the providers speaking the OpenAI chat
// completion API, which share its finish_reason vocabulary.
var openAICompatibleProviders = []struct {
	name   provider.Name
	create func(baseURL string) provider.OptionFunc
}{
	{openai.Name, func(baseURL string) provider.OptionFunc {
		return provider.WithChatCompletion(openai.Name, openai.Options{
			CommonOptions: provider.CommonOptions{BaseURL: baseURL, APIKey: "sk-test", Model: "gpt-test"},
		})
	}},
	{mistral.Name, func(baseURL string) provider.OptionFunc {
		return provider.WithChatCompletion(mistral.Name, mistral.Options{
			CommonOptions: provider.CommonOptions{BaseURL: baseURL, APIKey: "sk-test", Model: "mistral-small"},
		})
	}},
}

// TestChatCompletion_ReportsTheFinishReason asserts that an answer cut by the
// token limit says so, instead of passing for a complete one.
func TestChatCompletion_ReportsTheFinishReason(t *testing.T) {
	for _, p := range openAICompatibleProviders {
		t.Run(string(p.name), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hel"},"finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`))
			}))
			defer server.Close()

			client, err := provider.Create(context.Background(), p.create(server.URL))
			if err != nil {
				t.Fatalf("create: %v", err)
			}
			res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
			if err != nil {
				t.Fatalf("completion: %v", err)
			}
			if got := llm.FinishReasonOf(res); got != llm.FinishReasonLength {
				t.Errorf("finish reason = %q, want %q", got, llm.FinishReasonLength)
			}
		})
	}
}

// TestChatCompletionStream_ReportsTheFinishReason asserts that the finish
// reason survives the usage-only chunk OpenAI sends after it.
func TestChatCompletionStream_ReportsTheFinishReason(t *testing.T) {
	for _, p := range openAICompatibleProviders {
		t.Run(string(p.name), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte(contentChunk("Hel") + "\n\n" +
					`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}` + "\n\n" +
					`data: {"id":"1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}` + "\n\n" +
					"data: [DONE]\n\n"))
			}))
			defer server.Close()

			client, err := provider.Create(context.Background(), p.create(server.URL))
			if err != nil {
				t.Fatalf("create: %v", err)
			}
			chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
			if err != nil {
				t.Fatalf("stream: %v", err)
			}

			var finish llm.FinishReason
			for chunk := range chunks {
				if err := chunk.Error(); err != nil {
					t.Fatalf("stream chunk error: %v", err)
				}
				if chunk.IsComplete() {
					finish = llm.FinishReasonOf(chunk)
				}
			}
			if finish != llm.FinishReasonLength {
				t.Errorf("finish reason of the completion chunk = %q, want %q", finish, llm.FinishReasonLength)
			}
		})
	}
}

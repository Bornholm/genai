package openai_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/openai"
)

func TestChatCompletionStream_IgnoresSSEComments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		_, _ = w.Write([]byte(": PROCESSING\n\n" +
			": PROCESSING\r\n\r\n" +
			"data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n" +
			": comment inside an event\n\n" +
			"\n" +
			"data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n" +
			"data: [DONE]\n\n"))
	}))
	defer server.Close()

	client, err := provider.Create(context.Background(), provider.WithChatCompletion(openai.Name, openai.Options{
		CommonOptions: provider.CommonOptions{BaseURL: server.URL, APIKey: "sk-test", Model: "deepseek"},
	}))
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
}

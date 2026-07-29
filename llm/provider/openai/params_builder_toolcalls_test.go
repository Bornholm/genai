package openai

import (
	"context"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
)

// The assistant's narration around its tool calls must reach the provider.
// Without it the replayed history shows unexplained calls, and models re-plan
// and repeat calls they already made.
func TestConfigureMessagesKeepsToolCallsContent(t *testing.T) {
	opts := &llm.ChatCompletionOptions{
		Messages: []llm.Message{
			llm.NewToolCallsMessageWithContent(
				"Checking the config first.",
				llm.NewToolCall("call_1", "read_file", `{"path":"/etc/hosts"}`),
			),
		},
	}

	params := &openai.ChatCompletionNewParams{}
	if err := ConfigureMessages(context.Background(), opts, params); err != nil {
		t.Fatalf("could not configure messages: %v", err)
	}

	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	assistant := params.Messages[0].OfAssistant
	if assistant == nil {
		t.Fatal("expected an assistant message")
	}
	if got := assistant.Content.OfString.Value; got != "Checking the config first." {
		t.Errorf("content = %q, want the assistant narration", got)
	}
	if len(assistant.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(assistant.ToolCalls))
	}
}

// A tool calls message without narration must not gain an empty content field.
func TestConfigureMessagesOmitsEmptyToolCallsContent(t *testing.T) {
	opts := &llm.ChatCompletionOptions{
		Messages: []llm.Message{
			llm.NewToolCallsMessage(llm.NewToolCall("call_1", "ls", `{}`)),
		},
	}

	params := &openai.ChatCompletionNewParams{}
	if err := ConfigureMessages(context.Background(), opts, params); err != nil {
		t.Fatalf("could not configure messages: %v", err)
	}

	assistant := params.Messages[0].OfAssistant
	if assistant == nil {
		t.Fatal("expected an assistant message")
	}
	if got := assistant.Content.OfString.Value; got != "" {
		t.Errorf("content = %q, want empty", got)
	}
}

package conformance

import (
	"context"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

func testChatCompletion(t *testing.T, client llm.Client) {
	t.Helper()

	chatClient, ok := client.(llm.ChatCompletionClient)
	if !ok {
		t.Skip("client does not implement ChatCompletionClient")
	}

	ctx := context.Background()

	t.Run("BasicCompletion", func(t *testing.T) {
		res, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMessage(llm.RoleUser, "Say exactly the word: hello"),
			),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("ChatCompletion error: %v", err)
		}
		if res.Message().Content() == "" {
			t.Error("expected non-empty response content")
		}
		if res.Message().Role() != llm.RoleAssistant {
			t.Errorf("expected role %q, got %q", llm.RoleAssistant, res.Message().Role())
		}
		if res.Usage() == nil {
			t.Error("expected non-nil usage")
		}
	})

	t.Run("SystemPrompt", func(t *testing.T) {
		res, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMessage(llm.RoleSystem, "You are a bot that only ever replies with the single word PONG. Never say anything else."),
				llm.NewMessage(llm.RoleUser, "PING"),
			),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("ChatCompletion error: %v", err)
		}
		if !strings.Contains(strings.ToUpper(res.Message().Content()), "PONG") {
			t.Errorf("expected response to contain PONG, got: %q", res.Message().Content())
		}
	})

	t.Run("MultiTurn", func(t *testing.T) {
		res1, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMessage(llm.RoleUser, "My name is Alice. Just say OK."),
			),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("first turn error: %v", err)
		}

		res2, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMessage(llm.RoleUser, "My name is Alice. Just say OK."),
				res1.Message(),
				llm.NewMessage(llm.RoleUser, "What is my name? Reply with only the name."),
			),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("second turn error: %v", err)
		}
		if !strings.Contains(res2.Message().Content(), "Alice") {
			t.Errorf("expected response to contain 'Alice', got: %q", res2.Message().Content())
		}
	})
}

package conformance

import (
	"context"
	"testing"

	"github.com/bornholm/genai/llm"
)

func testReasoning(t *testing.T, client any) {
	t.Helper()

	chatClient, ok := client.(llm.ChatCompletionClient)
	if !ok {
		t.Skip("client does not implement ChatCompletionClient")
	}

	ctx := context.Background()

	t.Run("ReasoningTokensPresent", func(t *testing.T) {
		res, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMessage(llm.RoleUser, "What is 17 × 23? Show your work."),
			),
			llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortLow)),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("ChatCompletion error: %v", err)
		}

		rr, ok := res.(llm.ReasoningChatCompletionResponse)
		if !ok {
			t.Fatal("response does not implement ReasoningChatCompletionResponse")
		}
		if rr.Reasoning() == "" && len(rr.ReasoningDetails()) == 0 {
			t.Error("expected non-empty reasoning content or reasoning details")
		}
	})
}

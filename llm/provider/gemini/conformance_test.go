package gemini_test

import (
	"context"
	"os"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider"
	geminiProvider "github.com/bornholm/genai/llm/provider/gemini"
)

func TestConformance(t *testing.T) {
	apiKey := os.Getenv("CONFORMANCE_GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("CONFORMANCE_GEMINI_API_KEY not set")
	}

	chatModel := os.Getenv("CONFORMANCE_GEMINI_CHAT_MODEL")
	if chatModel == "" {
		chatModel = "gemini-2.5-flash"
	}

	ctx := context.Background()
	client, err := provider.Create(ctx, provider.WithChatCompletion(geminiProvider.Name, geminiProvider.Options{
		CommonOptions: provider.CommonOptions{
			BaseURL: os.Getenv("CONFORMANCE_GEMINI_BASE_URL"),
			APIKey:  apiKey,
			Model:   chatModel,
		},
	}))
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	conformance.New(client,
		conformance.WithFeatures(
			conformance.FeatureChatCompletion|
				conformance.FeatureStreaming|
				conformance.FeatureToolCalls|
				conformance.FeatureJSON|
				conformance.FeatureMultimodal|
				conformance.FeatureReasoning,
		),
	).Run(t)

	// Provider-specific: Gemini 3 refuses a function call turn replayed
	// without its thought signature, so the second turn of a tool loop
	// checks that the signatures make the round trip.
	t.Run("ReplaysThoughtSignaturesAcrossToolCalls", func(t *testing.T) {
		weather := llm.NewFuncTool("get_weather", "Get the current weather for a given city", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{"type": "string", "description": "The city name"},
			},
			"required": []string{"location"},
		}, nil)
		question := llm.NewMessage(llm.RoleUser, "What is the weather in Paris? Use the tool, then answer in one sentence.")
		reasoning := llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortLow))

		first, err := client.ChatCompletion(ctx, llm.WithMessages(question), llm.WithTools(weather), reasoning)
		if err != nil {
			t.Fatalf("first turn: %v", err)
		}
		if len(first.ToolCalls()) == 0 {
			t.Fatalf("expected a tool call, got %q", first.Message().Content())
		}

		var details []llm.ReasoningDetail
		var thoughts string
		if rr, ok := first.(llm.ReasoningChatCompletionResponse); ok {
			details, thoughts = rr.ReasoningDetails(), rr.Reasoning()
		}

		call := first.ToolCalls()[0]
		second, err := client.ChatCompletion(ctx,
			llm.WithMessages(
				question,
				llm.NewReasoningToolCallsMessageWithContent(first.Message().Content(), thoughts, details, first.ToolCalls()...),
				llm.NewToolMessage(call.ID(), llm.NewToolResult("18°C and sunny")),
			),
			llm.WithTools(weather),
			reasoning,
		)
		if err != nil {
			t.Fatalf("second turn: %v", err)
		}
		if second.Message().Content() == "" {
			t.Error("expected an answer on the second turn")
		}
	})
}

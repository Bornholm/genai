package responses_test

import (
	"context"
	"os"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider"
	responsesProvider "github.com/bornholm/genai/llm/provider/responses"
)

func TestConformance(t *testing.T) {
	apiKey := os.Getenv("CONFORMANCE_RESPONSES_API_KEY")
	if apiKey == "" {
		t.Skip("CONFORMANCE_RESPONSES_API_KEY not set")
	}

	chatModel := os.Getenv("CONFORMANCE_RESPONSES_CHAT_MODEL")
	if chatModel == "" {
		chatModel = "gpt-5-mini"
	}

	ctx := context.Background()
	client, err := provider.Create(ctx, provider.WithChatCompletion(responsesProvider.Name, responsesProvider.Options{
		CommonOptions: provider.CommonOptions{
			BaseURL: os.Getenv("CONFORMANCE_RESPONSES_BASE_URL"),
			APIKey:  apiKey,
			Model:   chatModel,
			Headers: conformance.HeadersFromEnv("CONFORMANCE_RESPONSES_HEADERS"),
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
		// OpenAI's reasoning models refuse any temperature unless their
		// reasoning is off.
		conformance.WithoutTemperature(),
	).Run(t)

	// Provider-specific: requests are not stored, so a reasoning model's
	// next turn only works if the encrypted reasoning of the previous one
	// is replayed.
	t.Run("ReplaysReasoningAcrossToolCalls", func(t *testing.T) {
		weather := llm.NewFuncTool("get_weather", "Get the current weather for a given city", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{"type": "string", "description": "The city name"},
			},
			"required": []string{"location"},
		}, nil)
		question := llm.NewMessage(llm.RoleUser, "Is it warmer in Paris than 17 × 23 / 20 degrees Celsius? Use the tool, then answer in one sentence.")
		reasoning := llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortMedium))

		first, err := client.ChatCompletion(ctx, llm.WithMessages(question), llm.WithTools(weather), reasoning)
		if err != nil {
			t.Fatalf("first turn: %v", err)
		}
		if len(first.ToolCalls()) == 0 {
			t.Fatalf("expected a tool call, got %q", first.Message().Content())
		}
		rr, ok := first.(llm.ReasoningChatCompletionResponse)
		if !ok {
			t.Fatal("response does not implement ReasoningChatCompletionResponse")
		}
		// A model may call the tool without reasoning first: there is
		// then nothing to replay.
		if len(rr.ReasoningDetails()) == 0 {
			t.Skip("the model did not reason before calling the tool")
		}

		call := first.ToolCalls()[0]
		second, err := client.ChatCompletion(ctx,
			llm.WithMessages(
				question,
				llm.NewReasoningToolCallsMessageWithContent(first.Message().Content(), rr.Reasoning(), rr.ReasoningDetails(), first.ToolCalls()...),
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

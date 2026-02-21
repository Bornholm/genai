package conformance

import (
	"context"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

var weatherTool = llm.NewFuncTool(
	"get_weather",
	"Get the current weather for a given city",
	map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{
				"type":        "string",
				"description": "The city name",
			},
		},
		"required": []string{"location"},
	},
	nil,
)

func testToolCalls(t *testing.T, client any) {
	t.Helper()

	chatClient, ok := client.(llm.ChatCompletionClient)
	if !ok {
		t.Skip("client does not implement ChatCompletionClient")
	}

	ctx := context.Background()

	t.Run("SingleToolCall", func(t *testing.T) {
		res, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMessage(llm.RoleUser, "What is the weather in Paris?"),
			),
			llm.WithTools(weatherTool),
			llm.WithToolChoice(llm.ToolChoiceRequired),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("ChatCompletion error: %v", err)
		}
		if len(res.ToolCalls()) == 0 {
			t.Fatal("expected at least one tool call")
		}
		tc := res.ToolCalls()[0]
		if tc.Name() != "get_weather" {
			t.Errorf("expected tool name %q, got %q", "get_weather", tc.Name())
		}
		params, ok := tc.Parameters().(string)
		if !ok {
			t.Fatalf("expected string parameters, got %T", tc.Parameters())
		}
		if !strings.Contains(strings.ToLower(params), "paris") {
			t.Errorf("expected parameters to contain 'paris', got: %q", params)
		}
	})

	t.Run("MultiTurnWithTool", func(t *testing.T) {
		res1, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMessage(llm.RoleUser, "What is the weather in London?"),
			),
			llm.WithTools(weatherTool),
			llm.WithToolChoice(llm.ToolChoiceRequired),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("first turn error: %v", err)
		}
		if len(res1.ToolCalls()) == 0 {
			t.Fatal("expected tool call in first turn")
		}
		tc := res1.ToolCalls()[0]

		toolCallsMsg := llm.NewToolCallsMessage(tc)
		toolResultMsg := llm.NewToolMessage(tc.ID(), llm.NewToolResult("Cloudy, 15°C"))

		res2, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMessage(llm.RoleUser, "What is the weather in London?"),
				toolCallsMsg,
				toolResultMsg,
			),
			llm.WithTools(weatherTool),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("second turn error: %v", err)
		}
		if res2.Message().Content() == "" {
			t.Error("expected non-empty content after tool result")
		}
	})
}

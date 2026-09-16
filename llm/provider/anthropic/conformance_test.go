package anthropic_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/conformance"
	"github.com/bornholm/genai/llm/provider"
	anthropicProvider "github.com/bornholm/genai/llm/provider/anthropic"
)

func TestConformance(t *testing.T) {
	apiKey := os.Getenv("CONFORMANCE_ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("CONFORMANCE_ANTHROPIC_API_KEY not set")
	}

	chatModel := os.Getenv("CONFORMANCE_ANTHROPIC_CHAT_MODEL")
	if chatModel == "" {
		chatModel = "claude-haiku-4-5"
	}

	ctx := context.Background()
	client, err := provider.Create(ctx,
		func(opts *provider.Options) error {
			opts.ChatCompletion = &provider.ResolvedClientOptions{
				Provider: anthropicProvider.Name,
				Specific: &anthropicProvider.Options{
					CommonOptions: provider.CommonOptions{
						APIKey: apiKey,
						Model:  chatModel,
					},
				},
			}
			return nil
		},
	)
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
				conformance.FeatureReasoning|
				conformance.FeatureCaching,
		),
	).Run(t)

	// Provider-specific: with extended thinking the API refuses
	// tool_choice any; the provider falls back to auto, and the model must
	// still call the tool when the prompt leaves it no other option.
	t.Run("RequiredToolChoiceWithReasoning", func(t *testing.T) {
		weather := llm.NewFuncTool("get_weather", "Get the current weather for a given city", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{"type": "string", "description": "The city name"},
			},
			"required": []string{"location"},
		}, nil)
		res, err := client.ChatCompletion(ctx,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "What is the weather in Paris? Use the tool.")),
			llm.WithTools(weather),
			llm.WithToolChoice(llm.ToolChoiceRequired),
			llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortLow)),
			llm.WithMaxCompletionTokens(4096),
		)
		if err != nil {
			t.Fatalf("ChatCompletion error: %v", err)
		}
		if len(res.ToolCalls()) == 0 || res.ToolCalls()[0].Name() != "get_weather" {
			t.Errorf("expected a get_weather tool call, got %v", res.ToolCalls())
		}
	})

	// Provider-specific: the Messages API documents structured output and
	// extended thinking as two independent features; this pins down that
	// the two can be combined on one request, since the provider does not
	// refuse the combination locally.
	t.Run("JSONWithReasoning", func(t *testing.T) {
		schema := llm.NewResponseSchema("answer", "An arithmetic answer", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"result": map[string]any{"type": "integer"},
			},
			"required":             []string{"result"},
			"additionalProperties": false,
		})
		res, err := client.ChatCompletion(ctx,
			llm.WithMessages(llm.NewMessage(llm.RoleUser, "What is 17 × 23? Reply with the JSON object only.")),
			llm.WithJSONResponse(schema),
			llm.WithReasoning(llm.NewReasoningOptions(llm.ReasoningEffortLow)),
			llm.WithMaxCompletionTokens(4096),
		)
		if err != nil {
			t.Fatalf("ChatCompletion error: %v", err)
		}
		var answer struct {
			Result int `json:"result"`
		}
		if err := json.Unmarshal([]byte(res.Message().Content()), &answer); err != nil {
			t.Fatalf("response is not valid JSON: %v\ncontent: %q", err, res.Message().Content())
		}
		if answer.Result != 391 {
			t.Errorf("expected 391, got %d", answer.Result)
		}
		if rr, ok := res.(llm.ReasoningChatCompletionResponse); !ok || (rr.Reasoning() == "" && len(rr.ReasoningDetails()) == 0) {
			t.Error("expected reasoning alongside the structured output")
		}
	})
}

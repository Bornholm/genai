package conformance

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

func testJSONResponse(t *testing.T, client llm.Client) {
	t.Helper()

	chatClient, ok := client.(llm.ChatCompletionClient)
	if !ok {
		t.Skip("client does not implement ChatCompletionClient")
	}

	ctx := context.Background()

	t.Run("SchemaConformance", func(t *testing.T) {
		schema := llm.NewResponseSchema(
			"person",
			"A person with a name and age",
			map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
				"required":             []string{"name", "age"},
				"additionalProperties": false,
			},
		)

		res, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMessage(llm.RoleUser, "Create a JSON object for a person named Alice who is 30 years old."),
			),
			llm.WithJSONResponse(schema),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("ChatCompletion error: %v", err)
		}

		var person struct {
			Name string `json:"name"`
			Age  int    `json:"age"`
		}
		if err := json.Unmarshal([]byte(res.Message().Content()), &person); err != nil {
			t.Fatalf("response is not valid JSON: %v\ncontent: %q", err, res.Message().Content())
		}
		if person.Name == "" {
			t.Error("expected non-empty name field in JSON response")
		}
		if person.Age <= 0 {
			t.Errorf("expected positive age in JSON response, got %d", person.Age)
		}
	})
}

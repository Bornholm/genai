package conformance

import (
	"context"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

func testStreaming(t *testing.T, client any) {
	t.Helper()

	streamClient, ok := client.(llm.ChatCompletionStreamingClient)
	if !ok {
		t.Skip("client does not implement ChatCompletionStreamingClient")
	}

	ctx := context.Background()

	t.Run("DeltasAccumulate", func(t *testing.T) {
		chunks, err := streamClient.ChatCompletionStream(ctx,
			llm.WithMessages(
				llm.NewMessage(llm.RoleUser, "Count from 1 to 5, one number per line, nothing else."),
			),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("ChatCompletionStream error: %v", err)
		}

		var buf strings.Builder
		var gotComplete bool

		for chunk := range chunks {
			if chunk.Error() != nil {
				t.Fatalf("stream chunk error: %v", chunk.Error())
			}
			if chunk.IsComplete() {
				gotComplete = true
				if chunk.Usage() == nil {
					t.Error("expected usage in final chunk")
				}
				continue
			}
			if d := chunk.Delta(); d != nil {
				buf.WriteString(d.Content())
			}
		}

		if !gotComplete {
			t.Error("stream did not emit a complete chunk")
		}
		content := buf.String()
		if content == "" {
			t.Error("accumulated content is empty")
		}
		for _, digit := range []string{"1", "2", "3", "4", "5"} {
			if !strings.Contains(content, digit) {
				t.Errorf("expected accumulated content to contain %q, got: %q", digit, content)
			}
		}
	})
}

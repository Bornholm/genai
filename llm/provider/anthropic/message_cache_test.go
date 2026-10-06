package anthropic

import (
	"testing"

	"github.com/bornholm/genai/llm"
)

func TestBuildMessages_ToolResultCarriesCacheControl(t *testing.T) {
	toolMessage := llm.NewToolMessage("toolu_01", llm.NewToolResult("done"))
	llm.SetCacheControl(toolMessage, &llm.CacheControl{Type: "ephemeral"})

	_, messages, err := buildMessages([]llm.Message{
		llm.NewMessage(llm.RoleUser, "Run it"),
		llm.NewToolCallsMessageWithContent("", llm.NewToolCall("toolu_01", "run", "{}")),
		toolMessage,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	last := messages[len(messages)-1]
	block := last.Content[len(last.Content)-1]
	if block.OfToolResult == nil {
		t.Fatalf("last block is not a tool_result: %#v", block)
	}
	if got := block.OfToolResult.CacheControl.Type; got != "ephemeral" {
		t.Errorf("tool_result cache control = %q, want ephemeral", got)
	}
}

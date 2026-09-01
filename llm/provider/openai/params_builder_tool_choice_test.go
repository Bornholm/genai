package openai

import (
	"context"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
)

// TestConfigureToolsNoToolsLeavesToolChoiceUnset covers a request carrying no
// tools at all.
//
// The default options set ToolChoice to "auto" (see llm.ChatCompletionOptions),
// so a caller that simply never declares a tool still arrives here with a
// choice. Sending "tool_choice" alongside an absent "tools" is rejected by
// OpenAI-compatible backends:
//
//	400 {"error":{"code":"invalid_request_error",
//	     "message":"'tool_choice' is only allowed when 'tools' are specified"}}
//
// Observed in production on 2026-09-01 behind a proxy relaying an agent's
// auxiliary calls — those carry no tools, and every one of them was rejected
// while the tool-carrying calls went through.
func TestConfigureToolsNoToolsLeavesToolChoiceUnset(t *testing.T) {
	params := &openai.ChatCompletionNewParams{}
	opts := &llm.ChatCompletionOptions{
		Messages:   []llm.Message{llm.NewMessage(llm.RoleUser, "hello")},
		ToolChoice: llm.ToolChoiceAuto,
	}

	if err := ConfigureTools(context.Background(), opts, params); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(params.Tools) != 0 {
		t.Fatalf("params.Tools: expected none, got %d", len(params.Tools))
	}

	if !params.ToolChoice.OfAuto.IsOmitted() {
		t.Fatalf("params.ToolChoice: expected it to stay unset when no tool is declared, got %v", params.ToolChoice.OfAuto)
	}
}

package openai

import (
	"context"
	"encoding/base64"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
)

// TestConfigureMessagesToolAttachments covers a tool answering with an image
// (an MCP server serving a stored screenshot): the "tool" role carries text
// only, so the medium must be relayed by a following user message rather than
// rejected — dropping it makes the model report it received nothing.
func TestConfigureMessagesToolAttachments(t *testing.T) {
	// 1x1 transparent GIF: small, and a media type every vision model accepts.
	data := base64.StdEncoding.EncodeToString([]byte{
		0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x80, 0x00,
		0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x21, 0xf9, 0x04, 0x01, 0x00,
		0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
		0x00, 0x02, 0x02, 0x44, 0x01, 0x00, 0x3b,
	})

	attachment, err := llm.NewImageAttachment("image/gif", data, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := llm.NewToolResult("[image attachment: image/gif, 43 bytes]", attachment)

	params := &openai.ChatCompletionNewParams{}
	opts := &llm.ChatCompletionOptions{
		Messages: []llm.Message{llm.NewToolMessage("call_1", result)},
	}

	if err := ConfigureMessages(context.Background(), opts, params); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if e, g := 2, len(params.Messages); e != g {
		t.Fatalf("params.Messages: expected %d messages (tool result + its media), got %d", e, g)
	}

	if params.Messages[0].OfTool == nil {
		t.Fatalf("params.Messages[0]: expected a tool message, got %+v", params.Messages[0])
	}

	if e, g := "call_1", params.Messages[0].OfTool.ToolCallID; e != g {
		t.Errorf("tool call id: expected %q, got %q", e, g)
	}

	user := params.Messages[1].OfUser
	if user == nil {
		t.Fatalf("params.Messages[1]: expected a user message carrying the media, got %+v", params.Messages[1])
	}

	parts := user.Content.OfArrayOfContentParts
	if e, g := 1, len(parts); e != g {
		t.Fatalf("user content parts: expected %d, got %d", e, g)
	}

	if parts[0].OfImageURL == nil {
		t.Errorf("user content part: expected an image, got %+v", parts[0])
	}
}

// A tool result without media keeps the plain single-message form.
func TestConfigureMessagesToolWithoutAttachments(t *testing.T) {
	params := &openai.ChatCompletionNewParams{}
	opts := &llm.ChatCompletionOptions{
		Messages: []llm.Message{llm.NewToolMessage("call_1", llm.NewToolResult("done"))},
	}

	if err := ConfigureMessages(context.Background(), opts, params); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if e, g := 1, len(params.Messages); e != g {
		t.Fatalf("params.Messages: expected %d, got %d", e, g)
	}
}

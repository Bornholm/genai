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

// A tool may return what the provider cannot carry, such as audio or a PDF:
// those attachments are left out, and the rest of the tool result and of the
// request goes through.
func TestConfigureMessagesToolAttachmentsTheProviderCannotCarry(t *testing.T) {
	image, err := llm.NewImageAttachment("image/png", "aGVsbG8=", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	audio, err := llm.NewAudioAttachment("audio/wav", "aGVsbG8=", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pdf, err := llm.NewBase64Attachment(llm.AttachmentTypeDocument, "application/pdf", "aGVsbG8=")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Run("some carried", func(t *testing.T) {
		params := &openai.ChatCompletionNewParams{}
		opts := &llm.ChatCompletionOptions{
			Messages: []llm.Message{llm.NewToolMessage("call_1", llm.NewToolResult("done", audio, image, pdf))},
		}

		if err := ConfigureMessages(context.Background(), opts, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if e, g := 2, len(params.Messages); e != g {
			t.Fatalf("params.Messages: expected %d messages (tool result + its image), got %d", e, g)
		}
		parts := params.Messages[1].OfUser.Content.OfArrayOfContentParts
		if len(parts) != 1 || parts[0].OfImageURL == nil {
			t.Errorf("user content parts: expected only the image, got %+v", parts)
		}
	})

	t.Run("none carried", func(t *testing.T) {
		params := &openai.ChatCompletionNewParams{}
		opts := &llm.ChatCompletionOptions{
			Messages: []llm.Message{llm.NewToolMessage("call_1", llm.NewToolResult("done", audio, pdf))},
		}

		if err := ConfigureMessages(context.Background(), opts, params); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if e, g := 1, len(params.Messages); e != g {
			t.Fatalf("params.Messages: expected only the tool result, got %d messages", g)
		}
		want := "done\n" + llm.OmittedAttachmentNote(audio) + "\n" + llm.OmittedAttachmentNote(pdf)
		if g := params.Messages[0].OfTool.Content.OfString.Value; g != want {
			t.Errorf("tool content: expected %q, got %q", want, g)
		}
	})
}

// Parallel tool calls: the assistant message must be followed by all its
// tool messages, so the media of every tool result go in one user message
// after the last of them.
func TestConfigureMessagesParallelToolResultsWithAttachments(t *testing.T) {
	image, err := llm.NewImageAttachment("image/png", "aGVsbG8=", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	params := &openai.ChatCompletionNewParams{}
	opts := &llm.ChatCompletionOptions{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, "Take both screenshots"),
			llm.NewToolCallsMessage(
				llm.NewToolCall("call_1", "screenshot", "{}"),
				llm.NewToolCall("call_2", "screenshot", "{}"),
			),
			llm.NewToolMessage("call_1", llm.NewToolResult("first", image)),
			llm.NewToolMessage("call_2", llm.NewToolResult("second", image)),
			llm.NewMessage(llm.RoleUser, "Compare them"),
		},
	}

	if err := ConfigureMessages(context.Background(), opts, params); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if e, g := 6, len(params.Messages); e != g {
		t.Fatalf("params.Messages: expected %d messages, got %d", e, g)
	}
	if params.Messages[2].OfTool == nil || params.Messages[3].OfTool == nil {
		t.Fatalf("expected the two tool messages right after the tool calls, got %+v and %+v", params.Messages[2], params.Messages[3])
	}
	media := params.Messages[4].OfUser
	if media == nil || len(media.Content.OfArrayOfContentParts) != 2 {
		t.Fatalf("expected one user message with both images after the tool messages, got %+v", params.Messages[4])
	}
	if params.Messages[5].OfUser == nil {
		t.Errorf("expected the next user message last, got %+v", params.Messages[5])
	}
}

// A user message keeps failing on what the provider cannot carry: the
// caller chose to send it.
func TestConfigureMessagesUserAttachmentTheProviderCannotCarry(t *testing.T) {
	audio, err := llm.NewAudioAttachment("audio/wav", "aGVsbG8=", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	params := &openai.ChatCompletionNewParams{}
	opts := &llm.ChatCompletionOptions{
		Messages: []llm.Message{llm.NewMultimodalMessage(llm.RoleUser, "Listen", audio)},
	}

	if err := ConfigureMessages(context.Background(), opts, params); err == nil {
		t.Fatal("expected an error for audio in a user message")
	}
}

package openrouter

import (
	"testing"

	"github.com/bornholm/genai/llm"
)

func audioAttachment(t *testing.T) llm.Attachment {
	t.Helper()

	attachment, err := llm.NewBase64Attachment(llm.AttachmentTypeAudio, "audio/wav", pngBase64)
	if err != nil {
		t.Fatalf("could not build attachment: %v", err)
	}
	return attachment
}

func TestBuildMessages_ToolAttachmentsFollowInUserMessage(t *testing.T) {
	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Look"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "screenshot", "{}")),
		llm.NewToolMessage("call_01", llm.NewToolResult("screenshot",
			pngAttachment(t), textAttachment(t, "Some notes."))),
		llm.NewMessage(llm.RoleUser, "What do you see?"),
	)

	if len(messages) != 5 {
		t.Fatalf("got %d messages, want 5: %#v", len(messages), messages)
	}

	tool := messages[2]
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_01" || tool["content"] != "screenshot" {
		t.Errorf("tool message = %#v, want the text alone", tool)
	}

	media := messages[3]
	if media["role"] != "user" {
		t.Fatalf("message after the tool result = %#v, want a user message", media)
	}
	assertParts(t, media, [][2]any{{"image_url", nil}, {"text", "Some notes."}})

	if messages[4]["content"] != "What do you see?" {
		t.Errorf("last message = %#v, want the user question", messages[4])
	}
}

func TestBuildMessages_ToolWithoutAttachments(t *testing.T) {
	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Run it"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "run", "{}")),
		llm.NewToolMessage("call_01", llm.NewToolResult("done")),
	)

	if len(messages) != 3 {
		t.Fatalf("got %d messages, want 3: %#v", len(messages), messages)
	}
	if messages[2]["content"] != "done" {
		t.Errorf("tool message = %#v, want the text alone", messages[2])
	}
}

func TestBuildMessages_ToolAttachmentsWithoutText(t *testing.T) {
	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Look"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "screenshot", "{}")),
		llm.NewToolMessage("call_01", llm.NewToolResult("", pngAttachment(t))),
	)

	content, _ := messages[2]["content"].(string)
	if content == "" {
		t.Errorf("tool message = %#v, want a placeholder text", messages[2])
	}
	assertParts(t, messages[3], [][2]any{{"image_url", nil}})
}

func TestBuildMessages_ToolAttachmentsTheProviderCannotCarry(t *testing.T) {
	audio := audioAttachment(t)

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Listen"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "record", "{}")),
		llm.NewToolMessage("call_01", llm.NewToolResult("recorded", audio, pngAttachment(t))),
	)

	if len(messages) != 4 {
		t.Fatalf("got %d messages, want 4: %#v", len(messages), messages)
	}

	want := "recorded\n" + llm.OmittedAttachmentNote(audio)
	if messages[2]["content"] != want {
		t.Errorf("tool content = %q, want %q", messages[2]["content"], want)
	}
	assertParts(t, messages[3], [][2]any{{"image_url", nil}})
}

func TestBuildMessages_ToolAttachmentsAllLeftOut(t *testing.T) {
	audio := audioAttachment(t)

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Listen"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "record", "{}")),
		llm.NewToolMessage("call_01", llm.NewToolResult("", audio)),
	)

	if len(messages) != 3 {
		t.Fatalf("got %d messages, want 3 (no media message): %#v", len(messages), messages)
	}
	if messages[2]["content"] != llm.OmittedAttachmentNote(audio) {
		t.Errorf("tool content = %q, want the note alone", messages[2]["content"])
	}
}

func TestBuildMessages_UserAttachmentTheProviderCannotCarry(t *testing.T) {
	_, err := buildMessages([]llm.Message{
		llm.NewMultimodalMessage(llm.RoleUser, "Listen", audioAttachment(t)),
	}, "anthropic/claude-sonnet-4.5")
	if err == nil {
		t.Fatal("expected an error for an audio attachment the caller chose to send")
	}
}

func TestBuildMessages_ParallelToolResultsWithAttachments(t *testing.T) {
	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Look at both"),
		llm.NewToolCallsMessage(
			llm.NewToolCall("call_01", "screenshot", "{}"),
			llm.NewToolCall("call_02", "screenshot", "{}"),
		),
		llm.NewToolMessage("call_01", llm.NewToolResult("first", pngAttachment(t))),
		llm.NewToolMessage("call_02", llm.NewToolResult("second", pngAttachment(t))),
	)

	if len(messages) != 5 {
		t.Fatalf("got %d messages, want 5: %#v", len(messages), messages)
	}
	if messages[2]["role"] != "tool" || messages[3]["role"] != "tool" {
		t.Fatalf("tool results are not adjacent: %#v", messages[2:4])
	}
	assertParts(t, messages[4], [][2]any{
		{"text", "Attachments of tool call call_01:"},
		{"image_url", nil},
		{"text", "Attachments of tool call call_02:"},
		{"image_url", nil},
	})
}

func TestBuildMessages_ParallelToolResultsOneWithAttachment(t *testing.T) {
	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Look"),
		llm.NewToolCallsMessage(
			llm.NewToolCall("call_01", "screenshot", "{}"),
			llm.NewToolCall("call_02", "run", "{}"),
		),
		llm.NewToolMessage("call_01", llm.NewToolResult("first", pngAttachment(t))),
		llm.NewToolMessage("call_02", llm.NewToolResult("done")),
	)

	// The label is kept: with two results the model cannot tell which one
	// the image belongs to.
	assertParts(t, messages[4], [][2]any{
		{"text", "Attachments of tool call call_01:"},
		{"image_url", nil},
	})
}

func TestBuildMessages_ToolAttachmentsCarryCacheControlAfterMedia(t *testing.T) {
	toolMessage := llm.NewToolMessage("call_01", llm.NewToolResult("screenshot", pngAttachment(t)))
	llm.SetCacheControl(toolMessage, ephemeral())

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Look"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "screenshot", "{}")),
		toolMessage,
	)

	if _, ok := messages[2]["content"].(string); !ok {
		t.Errorf("tool message content = %#v, want plain text without hint", messages[2]["content"])
	}
	if messages[3]["role"] != "user" {
		t.Fatalf("message after the tool result = %#v, want the media message", messages[3])
	}
	assertOnlyLastPartCached(t, messages[3])
}

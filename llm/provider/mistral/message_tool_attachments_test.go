package mistral

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

const pngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

func attachment(t *testing.T, kind llm.AttachmentType, mimeType string) llm.Attachment {
	t.Helper()

	a, err := llm.NewBase64Attachment(kind, mimeType, pngBase64)
	if err != nil {
		t.Fatalf("could not build attachment: %v", err)
	}
	return a
}

// wireMessages builds the messages of a request and decodes them back from
// their JSON form, so the assertions see what is sent.
func wireMessages(t *testing.T, msgs ...llm.Message) []map[string]any {
	t.Helper()

	params := buildParams(t, llm.WithMessages(msgs...))

	raw, err := json.Marshal(params.Messages)
	if err != nil {
		t.Fatalf("could not marshal messages: %v", err)
	}

	var decoded []map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("could not decode messages: %v", err)
	}
	return decoded
}

func partTypes(t *testing.T, message map[string]any) []any {
	t.Helper()

	parts, ok := message["content"].([]any)
	if !ok {
		t.Fatalf("%s message content is not a chunk list: %#v", message["role"], message["content"])
	}
	types := make([]any, len(parts))
	for i, part := range parts {
		types[i] = part.(map[string]any)["type"]
	}
	return types
}

func toolTurn(result llm.ToolResult) []llm.Message {
	return []llm.Message{
		llm.NewMessage(llm.RoleUser, "Look"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "screenshot", "{}")),
		llm.NewToolMessage("call_01", result),
	}
}

func TestToolResultImagesStayInToolMessage(t *testing.T) {
	image := attachment(t, llm.AttachmentTypeImage, "image/png")

	messages := wireMessages(t, toolTurn(llm.NewToolResult("screenshot", image, image))...)

	if len(messages) != 3 {
		t.Fatalf("got %d messages, want 3 (no user message after the tool result): %#v", len(messages), messages)
	}

	tool := messages[2]
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_01" {
		t.Fatalf("last message = %#v, want the tool result", tool)
	}
	types := partTypes(t, tool)
	want := []any{"text", "image_url", "image_url"}
	if len(types) != len(want) {
		t.Fatalf("chunk types = %v, want %v", types, want)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Errorf("chunk %d = %v, want %v", i, types[i], want[i])
		}
	}
	if text := tool["content"].([]any)[0].(map[string]any)["text"]; text != "screenshot" {
		t.Errorf("text chunk = %v, want the tool text", text)
	}
}

func TestToolResultImageWithoutText(t *testing.T) {
	image := attachment(t, llm.AttachmentTypeImage, "image/png")

	messages := wireMessages(t, toolTurn(llm.NewToolResult("", image))...)

	types := partTypes(t, messages[2])
	if len(types) != 1 || types[0] != "image_url" {
		t.Errorf("chunk types = %v, want the image alone", types)
	}
}

func TestToolResultWithoutAttachments(t *testing.T) {
	messages := wireMessages(t, toolTurn(llm.NewToolResult("done"))...)

	if messages[2]["content"] != "done" {
		t.Errorf("tool content = %#v, want the text alone", messages[2]["content"])
	}
}

func TestToolResultAttachmentsTheProviderCannotCarry(t *testing.T) {
	audio := attachment(t, llm.AttachmentTypeAudio, "audio/wav")
	pdf := attachment(t, llm.AttachmentTypeDocument, "application/pdf")
	image := attachment(t, llm.AttachmentTypeImage, "image/png")

	t.Run("all left out", func(t *testing.T) {
		messages := wireMessages(t, toolTurn(llm.NewToolResult("recorded", audio, pdf))...)

		want := "recorded\n" + llm.OmittedAttachmentNote(audio) + "\n" + llm.OmittedAttachmentNote(pdf)
		if messages[2]["content"] != want {
			t.Errorf("tool content = %#v, want %q", messages[2]["content"], want)
		}
	})

	t.Run("some left out", func(t *testing.T) {
		messages := wireMessages(t, toolTurn(llm.NewToolResult("recorded", audio, image))...)

		parts := messages[2]["content"].([]any)
		if len(parts) != 2 {
			t.Fatalf("chunks = %#v, want the text with its note, then the image", parts)
		}
		want := "recorded\n" + llm.OmittedAttachmentNote(audio)
		if text := parts[0].(map[string]any)["text"]; text != want {
			t.Errorf("text chunk = %v, want %q", text, want)
		}
		if parts[1].(map[string]any)["type"] != "image_url" {
			t.Errorf("chunk 1 = %#v, want the image", parts[1])
		}
	})
}

func TestUserAttachmentTheProviderCannotCarry(t *testing.T) {
	b := &paramsBuilder{model: "mistral-small-latest"}
	_, err := b.BuildParams(t.Context(), llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMultimodalMessage(llm.RoleUser, "Listen", attachment(t, llm.AttachmentTypeAudio, "audio/wav")),
	)))
	if err == nil {
		t.Fatal("expected an error for an audio attachment the caller chose to send")
	}
}

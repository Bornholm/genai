package mistral

import (
	"encoding/base64"
	"encoding/json"
	"strings"
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

func TestToolResultImagesFollowInLabelledUserMessage(t *testing.T) {
	image := attachment(t, llm.AttachmentTypeImage, "image/png")

	messages := wireMessages(t, append(toolTurn(llm.NewToolResult("screenshot", image, image)),
		llm.NewMessage(llm.RoleUser, "What do you see?"))...)

	if len(messages) != 5 {
		t.Fatalf("got %d messages, want 5: %#v", len(messages), messages)
	}

	tool := messages[2]
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_01" || tool["content"] != "screenshot" {
		t.Fatalf("tool message = %#v, want the text alone", tool)
	}

	media := messages[3]
	if media["role"] != "user" {
		t.Fatalf("message after the tool result = %#v, want a user message", media)
	}
	assertChunks(t, media, []any{"text", "image_url", "image_url"})
	if text := media["content"].([]any)[0].(map[string]any)["text"]; text != "Attachments of tool call call_01:" {
		t.Errorf("label = %v, want the tool call it comes from", text)
	}

	if messages[4]["content"] != "What do you see?" {
		t.Errorf("last message = %#v, want the user question", messages[4])
	}
}

func assertChunks(t *testing.T, message map[string]any, want []any) {
	t.Helper()

	types := partTypes(t, message)
	if len(types) != len(want) {
		t.Fatalf("chunk types = %v, want %v", types, want)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Errorf("chunk %d = %v, want %v", i, types[i], want[i])
		}
	}
}

func TestParallelToolResultsWithImages(t *testing.T) {
	image := attachment(t, llm.AttachmentTypeImage, "image/png")

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Look at both"),
		llm.NewToolCallsMessage(
			llm.NewToolCall("call_01", "screenshot", "{}"),
			llm.NewToolCall("call_02", "screenshot", "{}"),
		),
		llm.NewToolMessage("call_01", llm.NewToolResult("first", image)),
		llm.NewToolMessage("call_02", llm.NewToolResult("second", image)),
	)

	if len(messages) != 5 {
		t.Fatalf("got %d messages, want 5: %#v", len(messages), messages)
	}
	if messages[2]["role"] != "tool" || messages[3]["role"] != "tool" {
		t.Fatalf("tool results are not adjacent: %#v", messages[2:4])
	}
	assertChunks(t, messages[4], []any{"text", "image_url", "text", "image_url"})
}

func TestToolResultImageWithoutText(t *testing.T) {
	image := attachment(t, llm.AttachmentTypeImage, "image/png")

	messages := wireMessages(t, toolTurn(llm.NewToolResult("", image))...)

	if content, _ := messages[2]["content"].(string); content == "" {
		t.Errorf("tool message = %#v, want a placeholder text", messages[2])
	}
	assertChunks(t, messages[3], []any{"text", "image_url"})
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

		want := "recorded\n" + llm.OmittedAttachmentNote(audio)
		if messages[2]["content"] != want {
			t.Errorf("tool content = %#v, want %q", messages[2]["content"], want)
		}
		assertChunks(t, messages[3], []any{"text", "image_url"})
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

func documentAttachment(t *testing.T, text string) llm.Attachment {
	t.Helper()

	a, err := llm.NewBase64Attachment(llm.AttachmentTypeDocument, "text/plain", base64.StdEncoding.EncodeToString([]byte(text)))
	if err != nil {
		t.Fatalf("could not build attachment: %v", err)
	}
	return a
}

func TestToolResultImagesMistralCannotRead(t *testing.T) {
	tiff := attachment(t, llm.AttachmentTypeImage, "image/tiff")

	huge, err := llm.NewBase64Attachment(llm.AttachmentTypeImage, "image/png", strings.Repeat("AAAA", (maxImageBytes/3)+1))
	if err != nil {
		t.Fatalf("could not build attachment: %v", err)
	}

	for name, image := range map[string]llm.Attachment{"unsupported format": tiff, "over the size limit": huge} {
		t.Run(name, func(t *testing.T) {
			messages := wireMessages(t, toolTurn(llm.NewToolResult("captured", image))...)

			want := "captured\n" + llm.OmittedAttachmentNote(image)
			if messages[2]["content"] != want {
				t.Errorf("tool content = %#v, want %q", messages[2]["content"], want)
			}
		})
	}
}

func TestToolResultTextDocument(t *testing.T) {
	messages := wireMessages(t, toolTurn(llm.NewToolResult("read", documentAttachment(t, "Some notes.")))...)

	if messages[2]["content"] != "read" {
		t.Errorf("tool content = %#v, want the text alone", messages[2]["content"])
	}
	assertChunks(t, messages[3], []any{"text", "text"})
	if text := messages[3]["content"].([]any)[1].(map[string]any)["text"]; text != "Some notes." {
		t.Errorf("document chunk = %v, want its decoded text", text)
	}
}

func TestToolResultEmptyDocument(t *testing.T) {
	empty := rawDocument{}

	messages := wireMessages(t, toolTurn(llm.NewToolResult("read", empty))...)
	if messages[2]["content"] != "read" {
		t.Errorf("tool content = %#v, want the text alone", messages[2]["content"])
	}

	messages = wireMessages(t, toolTurn(llm.NewToolResult("", empty))...)
	if content, _ := messages[2]["content"].(string); content == "" {
		t.Errorf("tool content = %#v, want a placeholder text", messages[2]["content"])
	}
}

func TestUserTextDocument(t *testing.T) {
	messages := wireMessages(t, llm.NewMultimodalMessage(llm.RoleUser, "Summarize", documentAttachment(t, "Some notes.")))

	types := partTypes(t, messages[0])
	if len(types) != 2 || types[0] != "text" || types[1] != "text" {
		t.Errorf("chunk types = %v, want the text then the document", types)
	}
}

// rawDocument is an empty text document, which the attachment constructors
// refuse.
type rawDocument struct{}

func (rawDocument) Type() llm.AttachmentType     { return llm.AttachmentTypeDocument }
func (rawDocument) MimeType() string             { return "text/plain" }
func (rawDocument) Source() llm.AttachmentSource { return llm.AttachmentSourceBase64 }
func (rawDocument) Data() string                 { return "" }
func (rawDocument) ValidateFormat() error        { return nil }

func TestToolResultDocumentOverTheSizeLimit(t *testing.T) {
	big := documentAttachment(t, strings.Repeat("a", maxDocumentBytes+1))

	messages := wireMessages(t, toolTurn(llm.NewToolResult("read", big))...)

	want := "read\n" + llm.OmittedAttachmentNote(big)
	if messages[2]["content"] != want {
		t.Errorf("tool content = %#v, want %q", messages[2]["content"], want)
	}
}

func TestUserMessageWithOnlyEmptyDocuments(t *testing.T) {
	b := &paramsBuilder{model: "mistral-small-latest"}
	_, err := b.BuildParams(t.Context(), llm.NewChatCompletionOptions(llm.WithMessages(
		llm.NewMultimodalMessage(llm.RoleUser, "", rawDocument{}),
	)))
	if err == nil {
		t.Fatal("expected an error for a user message left without content")
	}
}

func TestDocumentAtTheSizeLimit(t *testing.T) {
	messages := wireMessages(t, llm.NewMultimodalMessage(llm.RoleUser, "Read", documentAttachment(t, strings.Repeat("a", maxDocumentBytes))))

	assertChunks(t, messages[0], []any{"text", "text"})
}

func TestJPGImageSentAsJPEG(t *testing.T) {
	for name, data := range map[string]string{
		"raw base64": pngBase64,
		"data URL":   "data:image/jpg;base64," + pngBase64,
	} {
		t.Run(name, func(t *testing.T) {
			image, err := llm.NewBase64Attachment(llm.AttachmentTypeImage, "image/jpg", data)
			if err != nil {
				t.Fatalf("could not build attachment: %v", err)
			}

			messages := wireMessages(t, llm.NewMultimodalMessage(llm.RoleUser, "Look", image))

			part := messages[0]["content"].([]any)[1].(map[string]any)
			url := part["image_url"].(map[string]any)["url"].(string)
			if url != "data:image/jpeg;base64,"+pngBase64 {
				t.Errorf("image url = %.40s, want an image/jpeg data URL", url)
			}
		})
	}
}

func TestSuccessiveToolRunsKeepTheirOwnImages(t *testing.T) {
	image := attachment(t, llm.AttachmentTypeImage, "image/png")

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Look twice"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "screenshot", "{}")),
		llm.NewToolMessage("call_01", llm.NewToolResult("first", image)),
		llm.NewToolCallsMessage(llm.NewToolCall("call_02", "screenshot", "{}")),
		llm.NewToolMessage("call_02", llm.NewToolResult("second", image)),
	)

	roles := make([]string, len(messages))
	for i, m := range messages {
		roles[i], _ = m["role"].(string)
	}
	if got := strings.Join(roles, " "); got != "user assistant tool user assistant tool user" {
		t.Fatalf("roles = %s, want each run followed by its images", got)
	}
	for i, id := range map[int]string{3: "call_01", 6: "call_02"} {
		assertChunks(t, messages[i], []any{"text", "image_url"})
		if text := messages[i]["content"].([]any)[0].(map[string]any)["text"]; text != "Attachments of tool call "+id+":" {
			t.Errorf("label of message %d = %v, want %s", i, text, id)
		}
	}
}

func TestEmptyToolResult(t *testing.T) {
	messages := wireMessages(t, toolTurn(llm.NewToolResult(""))...)

	if messages[2]["content"] != "[the tool returned no content]" {
		t.Errorf("tool content = %#v, want a placeholder text", messages[2]["content"])
	}
}

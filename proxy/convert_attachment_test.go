package proxy

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

// pngB64 is a minimal, valid base64 payload.
const pngB64 = "iVBORw0KGgo="

// TestExtractContentPartsDialects checks that every content-part encoding a
// client may realistically send yields the same attachment, so that an image
// is never silently lost between the client and the provider.
func TestExtractContentPartsDialects(t *testing.T) {
	for _, tc := range []struct {
		name string
		part string

		wantType   llm.AttachmentType
		wantSource llm.AttachmentSource
		wantMIME   string
		wantData   string
	}{
		{
			name:       "chat completions image_url data url",
			part:       `{"type":"image_url","image_url":{"url":"data:image/png;base64,` + pngB64 + `"}}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "image/png",
			wantData:   pngB64,
		},
		{
			name:       "chat completions image_url remote url",
			part:       `{"type":"image_url","image_url":{"url":"https://example.org/a.png","detail":"high"}}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceURL,
			wantMIME:   "image/png",
			wantData:   "https://example.org/a.png",
		},
		{
			name:       "remote url with query string",
			part:       `{"type":"image_url","image_url":{"url":"https://example.org/a.webp?sig=abc"}}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceURL,
			wantMIME:   "image/webp",
			wantData:   "https://example.org/a.webp?sig=abc",
		},
		{
			name:       "remote url without extension falls back",
			part:       `{"type":"image_url","image_url":{"url":"https://example.org/render"}}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceURL,
			wantMIME:   "image/jpeg",
			wantData:   "https://example.org/render",
		},
		{
			name:       "image_url as a bare string",
			part:       `{"type":"image_url","image_url":"data:image/png;base64,` + pngB64 + `"}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "image/png",
			wantData:   pngB64,
		},
		{
			name:       "responses api input_image",
			part:       `{"type":"input_image","image_url":"data:image/png;base64,` + pngB64 + `"}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "image/png",
			wantData:   pngB64,
		},
		{
			name:       "vercel ai sdk file part",
			part:       `{"type":"file","data":"` + pngB64 + `","mediaType":"image/png"}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "image/png",
			wantData:   pngB64,
		},
		{
			name:       "vercel ai sdk file part with data url",
			part:       `{"type":"file","data":"data:image/jpg;base64,` + pngB64 + `","mediaType":"image/jpg"}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "image/jpeg",
			wantData:   pngB64,
		},
		{
			name:       "anthropic block on the openai endpoint",
			part:       `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + pngB64 + `"}}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "image/png",
			wantData:   pngB64,
		},
		{
			name:       "responses api input_file",
			part:       `{"type":"input_file","filename":"notes.png","file_data":"data:image/png;base64,` + pngB64 + `"}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "image/png",
			wantData:   pngB64,
		},
		{
			name:       "file part nested under file key",
			part:       `{"type":"file","file":{"filename":"report.pdf","file_data":"data:application/pdf;base64,` + pngB64 + `"}}`,
			wantType:   llm.AttachmentTypeDocument,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "application/pdf",
			wantData:   pngB64,
		},
		{
			name:       "mime type inferred from filename",
			part:       `{"type":"file","filename":"shot.webp","data":"` + pngB64 + `"}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "image/webp",
			wantData:   pngB64,
		},
		{
			name:       "input_audio with format field",
			part:       `{"type":"input_audio","input_audio":{"data":"` + pngB64 + `","format":"wav"}}`,
			wantType:   llm.AttachmentTypeAudio,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "audio/wav",
			wantData:   pngB64,
		},
		{
			name:       "media type with parameters is normalised",
			part:       `{"type":"file","data":"` + pngB64 + `","mediaType":"image/PNG; charset=binary"}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "image/png",
			wantData:   pngB64,
		},
		{
			name:       "part without a type field is inferred",
			part:       `{"image_url":{"url":"data:image/png;base64,` + pngB64 + `"}}`,
			wantType:   llm.AttachmentTypeImage,
			wantSource: llm.AttachmentSourceBase64,
			wantMIME:   "image/png",
			wantData:   pngB64,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"look"},` + tc.part + `]}]}`

			_, _, chatOpts, err := ParseChatCompletionRequest(json.RawMessage(body))
			if err != nil {
				t.Fatalf("ParseChatCompletionRequest: %+v", err)
			}

			opts := llm.NewChatCompletionOptions(chatOpts...)
			if len(opts.Messages) != 1 {
				t.Fatalf("expected 1 message, got %d", len(opts.Messages))
			}

			msg := opts.Messages[0]
			if msg.Content() != "look" {
				t.Errorf("content: got %q, want %q", msg.Content(), "look")
			}

			atts := msg.Attachments()
			if len(atts) != 1 {
				t.Fatalf("expected 1 attachment, got %d", len(atts))
			}

			att := atts[0]
			if att.Type() != tc.wantType {
				t.Errorf("type: got %s, want %s", att.Type(), tc.wantType)
			}
			if att.Source() != tc.wantSource {
				t.Errorf("source: got %s, want %s", att.Source(), tc.wantSource)
			}
			if att.MimeType() != tc.wantMIME {
				t.Errorf("mime type: got %s, want %s", att.MimeType(), tc.wantMIME)
			}
			if att.Data() != tc.wantData {
				t.Errorf("data: got %q, want %q", att.Data(), tc.wantData)
			}
		})
	}
}

// TestExtractContentPartsTextOnly checks that parts carrying no attachment
// still contribute their text and never fail the request.
func TestExtractContentPartsTextOnly(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
		want    string
	}{
		{
			name:    "responses api input_text",
			content: `[{"type":"input_text","text":"hello"}]`,
			want:    "hello",
		},
		{
			// Becomes a text/plain document attachment, which providers inline
			// themselves; the message text stays empty.
			name:    "non base64 data url is re-encoded",
			content: `[{"type":"file","data":"data:text/plain,hello%20world","mediaType":"text/plain"}]`,
			want:    "",
		},
		{
			name:    "bare string part",
			content: `["hello"]`,
			want:    "hello",
		},
		{
			name:    "unknown part type is skipped",
			content: `[{"type":"text","text":"hi"},{"type":"video_url","video_url":{"url":"https://example.org/a.mp4"}}]`,
			want:    "hi",
		},
		{
			name:    "file with only a provider side id is skipped",
			content: `[{"type":"text","text":"hi"},{"type":"file","file":{"file_id":"file-abc"}}]`,
			want:    "hi",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"m","messages":[{"role":"user","content":` + tc.content + `}]}`

			_, _, chatOpts, err := ParseChatCompletionRequest(json.RawMessage(body))
			if err != nil {
				t.Fatalf("ParseChatCompletionRequest: %+v", err)
			}

			opts := llm.NewChatCompletionOptions(chatOpts...)
			if got := opts.Messages[0].Content(); got != tc.want {
				t.Errorf("content: got %q, want %q", got, tc.want)
			}
		})
	}
}

// TestExtractContentPartsMalformed checks that a recognised but broken part
// fails the request instead of degrading it into a text-only prompt.
func TestExtractContentPartsMalformed(t *testing.T) {
	for _, tc := range []struct{ name, content string }{
		{"data url without comma", `[{"type":"image_url","image_url":{"url":"data:image/png;base64"}}]`},
		{"base64 payload without media type", `[{"type":"file","data":"` + pngB64 + `"}]`},
		{"invalid base64", `[{"type":"image_url","image_url":{"url":"data:image/png;base64,not!base64"}}]`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"m","messages":[{"role":"user","content":` + tc.content + `}]}`
			if _, _, _, err := ParseChatCompletionRequest(json.RawMessage(body)); err == nil {
				t.Fatal("expected an error, got none")
			}
		})
	}
}

// TestAnthropicImageURLSource checks the /v1/messages surface, where a URL
// source used to be rejected outright for lack of a media type.
func TestAnthropicImageURLSource(t *testing.T) {
	msgs, err := ConvertAnthropicMessagesJSON(json.RawMessage(`[{"role":"user","content":[
		{"type":"text","text":"look"},
		{"type":"image","source":{"type":"url","url":"https://example.org/a.png"}}
	]}]`))
	if err != nil {
		t.Fatalf("ConvertAnthropicMessagesJSON: %+v", err)
	}
	if len(msgs) != 1 || len(msgs[0].Attachments()) != 1 {
		t.Fatalf("expected 1 message with 1 attachment, got %d message(s)", len(msgs))
	}
	if got := msgs[0].Attachments()[0].MimeType(); got != "image/png" {
		t.Errorf("mime type: got %s, want image/png", got)
	}
}

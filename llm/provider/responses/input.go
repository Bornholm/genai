package responses

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// reasoningFormat marks the reasoning details this provider produces and
// replays, with the value OpenRouter gives the same items.
const reasoningFormat = "openai-responses-v1"

// inputItem is one item of the request's input: a message, a reasoning item,
// a function call or a function call output. Fields not used by a type stay
// empty and are left out.
type inputItem struct {
	Type string `json:"type"`

	// message
	Role    string `json:"role,omitempty"`
	Content any    `json:"content,omitempty"`

	// reasoning
	ID               string        `json:"id,omitempty"`
	Summary          *[]summaryRef `json:"summary,omitempty"`
	EncryptedContent string        `json:"encrypted_content,omitempty"`

	// function_call and function_call_output
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    any    `json:"output,omitempty"`
}

type summaryRef struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// contentPart is one part of a message or function call output content.
type contentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
	FileData string `json:"file_data,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
	Filename string `json:"filename,omitempty"`
}

// buildInput converts genai messages to input items, in order. System
// messages stay where they are, as system role messages, rather than being
// hoisted into instructions: the API accepts them anywhere.
//
// Cache hints are dropped: the Responses API caches prompt prefixes on its
// own and takes no breakpoint.
func buildInput(msgs []llm.Message) ([]inputItem, error) {
	items := make([]inputItem, 0, len(msgs))

	for _, m := range msgs {
		switch m.Role() {
		case llm.RoleSystem:
			if len(m.Attachments()) > 0 {
				return nil, errors.New("system messages cannot have attachments")
			}
			if m.Content() == "" {
				continue
			}
			items = append(items, inputItem{Type: "message", Role: "system", Content: m.Content()})

		case llm.RoleUser:
			if len(m.Attachments()) == 0 {
				items = append(items, inputItem{Type: "message", Role: "user", Content: m.Content()})
				continue
			}
			parts := make([]contentPart, 0, 1+len(m.Attachments()))
			if content := m.Content(); content != "" {
				parts = append(parts, contentPart{Type: "input_text", Text: content})
			}
			for _, attachment := range m.Attachments() {
				part, err := attachmentPart(attachment)
				if err != nil {
					return nil, errors.WithStack(err)
				}
				parts = append(parts, part)
			}
			items = append(items, inputItem{Type: "message", Role: "user", Content: parts})

		case llm.RoleAssistant:
			if len(m.Attachments()) > 0 {
				return nil, errors.New("assistant messages cannot have attachments")
			}
			// An assistant turn without text has nothing to replay: its
			// reasoning items, without the output item they preceded, would
			// be refused.
			content := m.Content()
			if content == "" {
				continue
			}
			items = append(items, replayedReasoning(m)...)
			items = append(items, inputItem{Type: "message", Role: "assistant", Content: content})

		case llm.RoleToolCalls:
			if len(m.Attachments()) > 0 {
				return nil, errors.New("tool calls messages cannot have attachments")
			}
			toolCallsMessage, ok := m.(llm.ToolCallsMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool calls message type '%T'", m)
			}
			if m.Content() == "" && len(toolCallsMessage.ToolCalls()) == 0 {
				continue
			}
			items = append(items, replayedReasoning(m)...)
			if content := m.Content(); content != "" {
				items = append(items, inputItem{Type: "message", Role: "assistant", Content: content})
			}
			for _, tc := range toolCallsMessage.ToolCalls() {
				arguments, err := toolCallArguments(tc.Parameters())
				if err != nil {
					return nil, errors.Wrapf(err, "invalid parameters for tool call '%s'", tc.ID())
				}
				items = append(items, inputItem{
					Type:      "function_call",
					CallID:    tc.ID(),
					Name:      tc.Name(),
					Arguments: arguments,
				})
			}

		case llm.RoleTool:
			toolMessage, ok := m.(llm.ToolMessage)
			if !ok {
				return nil, errors.Errorf("unexpected tool message type '%T'", m)
			}
			items = append(items, inputItem{
				Type:   "function_call_output",
				CallID: toolMessage.ID(),
				Output: toolOutput(toolMessage),
			})

		default:
			return nil, errors.Errorf("unsupported message role '%s'", m.Role())
		}
	}

	return items, nil
}

// toolOutput builds a function call output: a plain string for a text
// result, a list of parts when the tool returned attachments. An attachment
// the API cannot carry is replaced by a note rather than failing the
// request: the caller did not choose what the tool returned.
func toolOutput(m llm.ToolMessage) any {
	if len(m.Attachments()) == 0 {
		return m.Content()
	}
	parts := make([]contentPart, 0, 1+len(m.Attachments()))
	if content := m.Content(); content != "" {
		parts = append(parts, contentPart{Type: "input_text", Text: content})
	}
	for i, attachment := range m.Attachments() {
		part, err := attachmentPart(attachment)
		if err != nil {
			slog.Warn("responses: leaving out a tool result attachment the provider cannot carry",
				slog.String("call_id", m.ID()), slog.Int("index", i),
				slog.String("type", string(attachment.Type())), slog.String("mime_type", attachment.MimeType()),
				slog.String("error", err.Error()))
			parts = append(parts, contentPart{Type: "input_text", Text: llm.OmittedAttachmentNote(attachment)})
			continue
		}
		parts = append(parts, part)
	}
	return parts
}

// replayedReasoning rebuilds the reasoning items of a previous turn from the
// details this provider produced. A reasoning item can only be replayed with
// its encrypted content, requests not being stored; details from another
// provider, or without encrypted content, are dropped.
func replayedReasoning(m llm.Message) []inputItem {
	rm, ok := m.(llm.ReasoningMessage)
	if !ok {
		return nil
	}

	var (
		items   []inputItem
		byID    = map[string]int{}
		details = rm.ReasoningDetails()
	)
	for _, d := range details {
		if d.Format != reasoningFormat || d.ID == "" {
			continue
		}
		index, seen := byID[d.ID]
		if !seen {
			index = len(items)
			byID[d.ID] = index
			items = append(items, inputItem{Type: "reasoning", ID: d.ID, Summary: &[]summaryRef{}})
		}
		switch d.Type {
		case llm.ReasoningDetailTypeEncrypted:
			items[index].EncryptedContent = d.Data
		case llm.ReasoningDetailTypeSummary:
			if d.Summary != "" {
				*items[index].Summary = append(*items[index].Summary, summaryRef{Type: "summary_text", Text: d.Summary})
			}
		}
	}

	replayable := items[:0]
	for _, item := range items {
		if item.EncryptedContent != "" {
			replayable = append(replayable, item)
		}
	}
	return replayable
}

// toolCallArguments renders the tool call parameters, kept as a JSON string
// by genai, as the arguments string of a function call. Arguments that are
// not a JSON object, as some models emit, are replayed as an empty object,
// as the openai provider does: the API would reject the whole request.
func toolCallArguments(parameters any) (string, error) {
	switch typed := parameters.(type) {
	case nil:
		return "{}", nil
	case string:
		return objectOrEmpty([]byte(typed)), nil
	case []byte:
		return objectOrEmpty(typed), nil
	default:
		raw, err := json.Marshal(typed)
		if err != nil {
			return "", errors.WithStack(err)
		}
		return string(raw), nil
	}
}

// attachmentPart converts an attachment to an input content part. Images go
// as input_image, PDF and text documents as input_file. Audio and video have
// no input part.
func attachmentPart(attachment llm.Attachment) (contentPart, error) {
	mimeType := normalizeMimeType(attachment.MimeType())

	switch attachment.Type() {
	case llm.AttachmentTypeImage:
		if !supportedImageMimeTypes[mimeType] {
			return contentPart{}, errors.Errorf("unsupported image MIME type: %s (supported: image/png, image/jpeg, image/gif, image/webp)", attachment.MimeType())
		}
		switch attachment.Source() {
		case llm.AttachmentSourceBase64:
			data, err := dataURL(mimeType, attachment.Data())
			if err != nil {
				return contentPart{}, err
			}
			return contentPart{Type: "input_image", ImageURL: data, Detail: "auto"}, nil
		case llm.AttachmentSourceURL:
			return contentPart{Type: "input_image", ImageURL: attachment.Data(), Detail: "auto"}, nil
		}

	case llm.AttachmentTypeDocument:
		if mimeType != "application/pdf" && !strings.HasPrefix(mimeType, "text/") {
			return contentPart{}, errors.Errorf("unsupported document MIME type: %s (supported: application/pdf, text/*)", attachment.MimeType())
		}
		switch attachment.Source() {
		case llm.AttachmentSourceBase64:
			data, err := dataURL(mimeType, attachment.Data())
			if err != nil {
				return contentPart{}, err
			}
			return contentPart{Type: "input_file", FileData: data, Filename: documentFilename(mimeType)}, nil
		case llm.AttachmentSourceURL:
			return contentPart{Type: "input_file", FileURL: attachment.Data()}, nil
		}
	}

	return contentPart{}, errors.Errorf("unsupported attachment type '%s' (source '%s')", attachment.Type(), attachment.Source())
}

// documentFilename names an inline document: the API requires a filename
// with file_data and genai attachments carry none.
func documentFilename(mimeType string) string {
	if mimeType == "application/pdf" {
		return "document.pdf"
	}
	return "document.txt"
}

// dataURL returns data as a base64 data URL, which is what input_image and
// input_file take for inline content.
func dataURL(mimeType, data string) (string, error) {
	if !strings.HasPrefix(data, "data:") {
		if _, err := base64.StdEncoding.DecodeString(data); err != nil {
			return "", errors.Wrap(err, "invalid base64 attachment")
		}
		return fmt.Sprintf("data:%s;base64,%s", mimeType, data), nil
	}
	header, _, found := strings.Cut(data, ",")
	if !found {
		return "", errors.New("malformed data URL: no payload")
	}
	if !strings.HasSuffix(header, ";base64") {
		return "", errors.New("unsupported data URL: only base64-encoded payloads are accepted")
	}
	return data, nil
}

// normalizeMimeType lowercases a MIME type, drops its parameters and maps
// the common image/jpg misspelling to its registered name.
func normalizeMimeType(mimeType string) string {
	base, _, _ := strings.Cut(mimeType, ";")
	base = strings.ToLower(strings.TrimSpace(base))
	if base == "image/jpg" {
		return "image/jpeg"
	}
	return base
}

var supportedImageMimeTypes = map[string]bool{
	"image/png":  true,
	"image/jpeg": true,
	"image/gif":  true,
	"image/webp": true,
}

// objectOrEmpty returns raw when it is a JSON object, an empty object
// otherwise.
func objectOrEmpty(raw []byte) string {
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil || object == nil {
		return "{}"
	}
	return string(raw)
}

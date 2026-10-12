package mistral

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/pkg/errors"
)

// ConfigureMistralMessages configures messages for Mistral API.
// It handles reasoning messages by passing the content through.
// For reasoning models, Mistral handles the structured content automatically.
func ConfigureMistralMessages(ctx context.Context, opts *llm.ChatCompletionOptions, params *openai.ChatCompletionNewParams) error {
	// For Mistral, we can use the standard OpenAI message configuration
	// but need to handle reasoning messages specially
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(opts.Messages))

	// The media of tool results go in a single user message after the last
	// of a run of tool messages, as the openai provider does: an assistant
	// message with parallel tool calls must be followed by all its tool
	// messages, uninterrupted. Each result's media are preceded by a label
	// naming its tool call: against the live API, mistral-medium-latest
	// timed out on a user message holding an image alone, and on images
	// inside the tool message, while it read a labelled image correctly.
	// A model without vision, such as codestral-latest, answers 400 to such
	// a request: the caller picks the model, so tools returning images call
	// for one that reads them.
	var toolMedia []openai.ChatCompletionContentPartUnionParam
	flushToolMedia := func() {
		if len(toolMedia) > 0 {
			messages = append(messages, openai.UserMessage(toolMedia))
		}
		toolMedia = nil
	}

	for _, m := range opts.Messages {
		if m.Role() != llm.RoleTool {
			flushToolMedia()
		}

		switch m.Role() {
		case llm.RoleSystem:
			messages = append(messages, openai.SystemMessage(m.Content()))

		case llm.RoleUser:
			if len(m.Attachments()) > 0 {
				// Handle multimodal user message
				contentParts := make([]openai.ChatCompletionContentPartUnionParam, 0)

				// Add text content if present
				if m.Content() != "" {
					contentParts = append(contentParts, openai.ChatCompletionContentPartUnionParam{
						OfText: &openai.ChatCompletionContentPartTextParam{
							Text: m.Content(),
						},
					})
				}

				// Add attachments using OpenAI types
				for _, attachment := range m.Attachments() {
					part, err := convertAttachment(attachment)
					if err != nil {
						return errors.Wrapf(err, "failed to convert attachment")
					}
					if isEmptyText(part) {
						continue
					}
					contentParts = append(contentParts, part)
				}

				if len(contentParts) == 0 {
					// Only empty documents, and no text: an empty user
					// message would be refused upstream with a less clear
					// error.
					return errors.Errorf("user message has no content: no text, and its attachments are empty documents")
				}

				messages = append(messages, openai.UserMessage(contentParts))
			} else {
				messages = append(messages, openai.UserMessage(m.Content()))
			}

		case llm.RoleAssistant:
			if len(m.Attachments()) > 0 {
				return errors.Errorf("assistant messages cannot have attachments")
			}
			// For reasoning messages, pass through the content
			// The reasoning tokens will be handled by the API
			messages = append(messages, openai.AssistantMessage(m.Content()))

		case llm.RoleTool:
			toolMessage, ok := m.(llm.ToolMessage)
			if !ok {
				return errors.Errorf("unexpected tool message type '%T'", m)
			}

			message, parts := toolResultMessage(ctx, toolMessage)
			messages = append(messages, message)
			if len(parts) > 0 {
				toolMedia = append(toolMedia, openai.TextContentPart("Attachments of tool call "+toolMessage.ID()+":"))
				toolMedia = append(toolMedia, parts...)
			}

		case llm.RoleToolCalls:
			toolCallsMessage, ok := m.(llm.ToolCallsMessage)
			if !ok {
				return errors.Errorf("unexpected tool calls message type '%T'", m)
			}

			message := openai.ChatCompletionAssistantMessageParam{}

			toolCalls := make([]openai.ChatCompletionMessageToolCallParam, 0, len(toolCallsMessage.ToolCalls()))
			for _, tc := range toolCallsMessage.ToolCalls() {
				arguments, ok := tc.Parameters().(string)
				if !ok {
					return errors.Errorf("expected string parameters for tool call %s, got %T", tc.ID(), tc.Parameters())
				}

				toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
					ID: tc.ID(),
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      tc.Name(),
						Arguments: arguments,
					},
				})
			}
			message.ToolCalls = toolCalls

			messages = append(messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &message})

		default:
			return errors.Errorf("unsupported message role: %s", m.Role())
		}
	}

	flushToolMedia()

	params.Messages = messages

	return nil
}

// toolResultMessage builds the message of a tool result, text only, and
// returns apart the parts of its attachments, which go in the user message
// after the run of tool messages.
//
// What the provider cannot carry, such as audio or a document other than
// text, is left out rather than failing the whole request: the caller did
// not choose what the tool returned. A note appended to the tool's text
// tells the model something was there.
func toolResultMessage(ctx context.Context, toolMessage llm.ToolMessage) (openai.ChatCompletionMessageParamUnion, []openai.ChatCompletionContentPartUnionParam) {
	content := toolMessage.Content()
	var parts []openai.ChatCompletionContentPartUnionParam

	for i, attachment := range toolMessage.Attachments() {
		part, err := convertAttachment(attachment)
		if err != nil {
			slog.WarnContext(ctx, "mistral: leaving out a tool result attachment the provider cannot carry",
				slog.String("tool_call_id", toolMessage.ID()), slog.Int("index", i),
				slog.String("type", string(attachment.Type())), slog.String("mime_type", attachment.MimeType()),
				slog.String("error", err.Error()))
			if content != "" {
				content += "\n"
			}
			content += llm.OmittedAttachmentNote(attachment)
			continue
		}
		if isEmptyText(part) {
			continue
		}
		parts = append(parts, part)
	}

	if content == "" && len(toolMessage.Attachments()) > 0 {
		if len(parts) > 0 {
			content = "[the attachments of this result follow in the next message]"
		} else {
			// Its only attachments were empty documents.
			content = "[the tool returned no content]"
		}
	}

	return openai.ToolMessage(content, toolMessage.ID()), parts
}

// convertAttachment converts an llm.Attachment to OpenAI content part format
func convertAttachment(attachment llm.Attachment) (openai.ChatCompletionContentPartUnionParam, error) {
	switch attachment.Type() {
	case llm.AttachmentTypeImage:
		return convertImageAttachment(attachment)
	case llm.AttachmentTypeDocument:
		return convertDocumentAttachment(attachment)
	default:
		return openai.ChatCompletionContentPartUnionParam{}, errors.Errorf("unsupported attachment type: %s", attachment.Type())
	}
}

// supportedImageMimeTypes are the image formats Mistral vision models
// accept; an animated GIF is refused upstream but cannot be told apart here.
var supportedImageMimeTypes = map[string]struct{}{
	"image/png":  {},
	"image/jpeg": {},
	"image/jpg":  {},
	"image/webp": {},
	"image/gif":  {},
}

// maxImageBytes is the size limit Mistral sets on an image.
const maxImageBytes = 10 * 1024 * 1024

// convertImageAttachment converts an image attachment to OpenAI image content
// part, refusing a format or a size Mistral would reject: a tool result's
// image is then left out instead of failing the request.
func convertImageAttachment(attachment llm.Attachment) (openai.ChatCompletionContentPartUnionParam, error) {
	if _, ok := supportedImageMimeTypes[strings.ToLower(attachment.MimeType())]; !ok {
		return openai.ChatCompletionContentPartUnionParam{}, errors.Errorf("unsupported image MIME type: %s (supported: png, jpeg, webp, gif)", attachment.MimeType())
	}

	switch attachment.Source() {
	case llm.AttachmentSourceBase64:
		data := attachment.Data()
		if size := base64.StdEncoding.DecodedLen(len(base64Payload(data))); size > maxImageBytes {
			return openai.ChatCompletionContentPartUnionParam{}, errors.Errorf("image of about %d bytes exceeds the %d bytes limit", size, maxImageBytes)
		}
		if len(data) > 0 && !hasDataPrefix(data) {
			data = "data:" + attachment.MimeType() + ";base64," + data
		}

		return openai.ChatCompletionContentPartUnionParam{
			OfImageURL: &openai.ChatCompletionContentPartImageParam{
				ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
					URL: data,
				},
			},
		}, nil

	case llm.AttachmentSourceURL:
		return openai.ChatCompletionContentPartUnionParam{
			OfImageURL: &openai.ChatCompletionContentPartImageParam{
				ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
					URL: attachment.Data(),
				},
			},
		}, nil

	default:
		return openai.ChatCompletionContentPartUnionParam{}, errors.Errorf("unsupported attachment source: %s", attachment.Source())
	}
}

// maxDocumentBytes caps the decoded size of a text document, as the
// openrouter provider does: a tool result's larger document is left out
// rather than filling the context or failing the request.
const maxDocumentBytes = 1 * 1024 * 1024

// convertDocumentAttachment converts a text/* document to a text content
// part holding its decoded content. Other documents are not supported.
func convertDocumentAttachment(attachment llm.Attachment) (openai.ChatCompletionContentPartUnionParam, error) {
	if !strings.HasPrefix(strings.ToLower(attachment.MimeType()), "text/") {
		return openai.ChatCompletionContentPartUnionParam{}, errors.Errorf("unsupported document MIME type: %s (only text/* types are supported)", attachment.MimeType())
	}
	if attachment.Source() != llm.AttachmentSourceBase64 {
		return openai.ChatCompletionContentPartUnionParam{}, errors.Errorf("unsupported document source: %s (only base64 is supported)", attachment.Source())
	}

	decoded, err := base64.StdEncoding.DecodeString(base64Payload(attachment.Data()))
	if err != nil {
		return openai.ChatCompletionContentPartUnionParam{}, errors.Wrap(err, "failed to decode base64 document content")
	}
	if len(decoded) > maxDocumentBytes {
		return openai.ChatCompletionContentPartUnionParam{}, errors.Errorf("document of %d bytes exceeds the %d bytes limit", len(decoded), maxDocumentBytes)
	}

	return openai.TextContentPart(string(decoded)), nil
}

// isEmptyText reports whether part is a text part without text, as an empty
// document gives: Mistral may refuse it, and nothing is lost leaving it out.
func isEmptyText(part openai.ChatCompletionContentPartUnionParam) bool {
	return part.OfText != nil && part.OfText.Text == ""
}

// base64Payload returns data without its data URL prefix, if any.
func base64Payload(data string) string {
	if hasDataPrefix(data) {
		if _, payload, ok := strings.Cut(data, ","); ok {
			return payload
		}
	}
	return data
}

func hasDataPrefix(s string) bool {
	return len(s) > 7 && s[:5] == "data:"
}

// Content part type constants for parsing
const (
	contentPartTypeText     = "text"
	contentPartTypeThinking = "thinking"
)

// extractThinkingFromResponse extracts thinking content from Mistral's response.
// The response content can be either a string or a structured array with thinking blocks.
func extractThinkingFromResponse(content any) (string, []llm.ReasoningDetail, string) {
	if content == nil {
		return "", nil, ""
	}

	// Handle string content
	if str, ok := content.(string); ok {
		// Check if the string contains JSON (structured content from Mistral)
		// Mistral returns thinking blocks as JSON array when using structured content
		trimmed := strings.TrimSpace(str)
		if len(trimmed) > 0 && trimmed[0] == '[' {
			// Try to parse as JSON array
			var parts []map[string]any
			if err := json.Unmarshal([]byte(str), &parts); err == nil {
				return parseContentPartsMap(parts)
			}
		}
		// Plain string content (old format without thinking)
		return str, nil, ""
	}

	// Handle array of content parts (new format with thinking blocks)
	if parts, ok := content.([]any); ok {
		return parseContentParts(parts)
	}

	// Handle []map[string]any format
	if parts, ok := content.([]map[string]any); ok {
		return parseContentPartsMap(parts)
	}

	// Fallback: convert to string
	return "", nil, ""
}

func parseContentParts(parts []any) (string, []llm.ReasoningDetail, string) {
	var textContent string
	var reasoning string
	details := make([]llm.ReasoningDetail, 0)

	for i, part := range parts {
		partMap, ok := part.(map[string]any)
		if !ok {
			continue
		}

		partType, _ := partMap["type"].(string)

		switch partType {
		case contentPartTypeText:
			if text, ok := partMap["text"].(string); ok {
				textContent += text
			}
		case contentPartTypeThinking:
			if thinking, ok := partMap["thinking"]; ok {
				thinkingDetails := parseThinkingContent(thinking, i)
				details = append(details, thinkingDetails...)
			}
		}
	}

	// Build reasoning string from details
	for _, d := range details {
		reasoning += d.Text
	}

	return textContent, details, reasoning
}

func parseContentPartsMap(parts []map[string]any) (string, []llm.ReasoningDetail, string) {
	var textContent string
	var reasoning string
	details := make([]llm.ReasoningDetail, 0)

	for i, part := range parts {
		partType, _ := part["type"].(string)

		switch partType {
		case contentPartTypeText:
			if text, ok := part["text"].(string); ok {
				textContent += text
			}
		case contentPartTypeThinking:
			if thinking, ok := part["thinking"]; ok {
				thinkingDetails := parseThinkingContent(thinking, i)
				details = append(details, thinkingDetails...)
			}
		}
	}

	// Build reasoning string from details
	for _, d := range details {
		reasoning += d.Text
	}

	return textContent, details, reasoning
}

func parseThinkingContent(thinking any, index int) []llm.ReasoningDetail {
	details := make([]llm.ReasoningDetail, 0)

	// Handle array of thinking blocks
	if blocks, ok := thinking.([]any); ok {
		for _, block := range blocks {
			if detail := parseThinkingBlock(block, index); detail != nil {
				details = append(details, *detail)
			}
		}
	} else if blocks, ok := thinking.([]map[string]any); ok {
		for _, block := range blocks {
			if detail := parseThinkingBlockMap(block, index); detail != nil {
				details = append(details, *detail)
			}
		}
	}

	return details
}

func parseThinkingBlock(block any, index int) *llm.ReasoningDetail {
	blockMap, ok := block.(map[string]any)
	if !ok {
		return nil
	}

	blockType, _ := blockMap["type"].(string)

	switch blockType {
	case "text":
		if text, ok := blockMap["text"].(string); ok {
			return &llm.ReasoningDetail{
				ID:    "",
				Type:  llm.ReasoningDetailTypeText,
				Text:  text,
				Index: index,
			}
		}
	case "encrypted":
		if data, ok := blockMap["encrypted"].(string); ok {
			return &llm.ReasoningDetail{
				ID:    "",
				Type:  llm.ReasoningDetailTypeEncrypted,
				Data:  data,
				Index: index,
			}
		}
	}

	return nil
}

func parseThinkingBlockMap(block map[string]any, index int) *llm.ReasoningDetail {
	blockType, _ := block["type"].(string)

	switch blockType {
	case "text":
		if text, ok := block["text"].(string); ok {
			return &llm.ReasoningDetail{
				ID:    "",
				Type:  llm.ReasoningDetailTypeText,
				Text:  text,
				Index: index,
			}
		}
	case "encrypted":
		if data, ok := block["encrypted"].(string); ok {
			return &llm.ReasoningDetail{
				ID:    "",
				Type:  llm.ReasoningDetailTypeEncrypted,
				Data:  data,
				Index: index,
			}
		}
	}

	return nil
}

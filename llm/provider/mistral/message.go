package mistral

import (
	"context"
	"encoding/json"
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

	for _, m := range opts.Messages {
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
					contentParts = append(contentParts, part)
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

			// Tool messages only support text content for now
			messages = append(messages, openai.ToolMessage(toolMessage.Content(), toolMessage.ID()))

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

	params.Messages = messages

	return nil
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

// convertImageAttachment converts an image attachment to OpenAI image content part
func convertImageAttachment(attachment llm.Attachment) (openai.ChatCompletionContentPartUnionParam, error) {
	switch attachment.Source() {
	case llm.AttachmentSourceBase64:
		data := attachment.Data()
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

// convertDocumentAttachment converts a document attachment to OpenAI text content part
func convertDocumentAttachment(attachment llm.Attachment) (openai.ChatCompletionContentPartUnionParam, error) {
	return openai.ChatCompletionContentPartUnionParam{}, errors.Errorf("document attachments not yet supported for Mistral")
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

package openrouter

import (
	"fmt"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"github.com/revrost/go-openrouter"
)

// OpenRouterAttachmentValidator implements provider-specific validation for OpenRouter
type OpenRouterAttachmentValidator struct {
	model string
}

// ValidateAttachment implements llm.AttachmentValidator - Layer 2 validation
func (v *OpenRouterAttachmentValidator) ValidateAttachment(attachment llm.Attachment) error {

	// Validate attachment type support
	switch attachment.Type() {
	case llm.AttachmentTypeImage:
		return v.validateImageAttachment(attachment)
	case llm.AttachmentTypeAudio:
		return llm.NewAttachmentError("provider", "type", "audio attachments support depends on the underlying model via OpenRouter")
	case llm.AttachmentTypeVideo:
		return llm.NewAttachmentError("provider", "type", "video attachments support depends on the underlying model via OpenRouter")
	case llm.AttachmentTypeDocument:
		return llm.NewAttachmentError("provider", "type", "document attachments support depends on the underlying model via OpenRouter")
	default:
		return llm.NewAttachmentError("provider", "type", fmt.Sprintf("unsupported attachment type: %s", attachment.Type()))
	}
}

// validateImageAttachment validates image-specific constraints for OpenRouter
func (v *OpenRouterAttachmentValidator) validateImageAttachment(attachment llm.Attachment) error {
	// OpenRouter generally supports common image formats
	supportedMimeTypes := []string{
		"image/png",
		"image/jpeg",
		"image/webp",
		"image/gif",
	}

	mimeType := strings.ToLower(attachment.MimeType())
	supported := false
	for _, supportedType := range supportedMimeTypes {
		if mimeType == supportedType {
			supported = true
			break
		}
	}

	if !supported {
		return llm.NewAttachmentError("provider", "mime_type", fmt.Sprintf("potentially unsupported image MIME type: %s (commonly supported: %v)", attachment.MimeType(), supportedMimeTypes))
	}

	// For base64 attachments, apply reasonable size limits
	if attachment.Source() == llm.AttachmentSourceBase64 {
		data := attachment.Data()
		// Remove data URL prefix if present
		if strings.HasPrefix(data, "data:") {
			parts := strings.SplitN(data, ",", 2)
			if len(parts) == 2 {
				data = parts[1]
			}
		}

		// Apply a conservative size limit (varies by underlying model)
		maxBase64Size := 10 * 1024 * 1024 * 4 / 3 // ~13.3MB in base64
		if len(data) > maxBase64Size {
			return llm.NewAttachmentError("provider", "size", fmt.Sprintf("image size may exceed limits for some models (estimated size: %d bytes)", len(data)*3/4))
		}
	}

	return nil
}

// NewOpenRouterAttachmentValidator creates a new OpenRouter attachment validator
func NewOpenRouterAttachmentValidator(model string) *OpenRouterAttachmentValidator {
	return &OpenRouterAttachmentValidator{
		model: model,
	}
}

var _ llm.AttachmentValidator = &OpenRouterAttachmentValidator{}

// ConvertAttachmentToContent converts an attachment to OpenRouter content format
func ConvertAttachmentToContent(attachment llm.Attachment, textContent string) (openrouter.Content, error) {
	switch attachment.Type() {
	case llm.AttachmentTypeImage:
		return convertImageAttachment(attachment, textContent)
	default:
		return openrouter.Content{}, errors.Errorf("unsupported attachment type for conversion: %s", attachment.Type())
	}
}

// convertImageAttachment converts an image attachment to OpenRouter content format
func convertImageAttachment(attachment llm.Attachment, textContent string) (openrouter.Content, error) {
	parts := make([]openrouter.ChatMessagePart, 0)

	// Add text content if present
	if textContent != "" {
		parts = append(parts, openrouter.ChatMessagePart{
			Type: openrouter.ChatMessagePartTypeText,
			Text: textContent,
		})
	}

	// Add image part
	switch attachment.Source() {
	case llm.AttachmentSourceBase64:
		// For base64, format as data URL if not already
		data := attachment.Data()
		if !strings.HasPrefix(data, "data:") {
			data = fmt.Sprintf("data:%s;base64,%s", attachment.MimeType(), data)
		}

		parts = append(parts, openrouter.ChatMessagePart{
			Type: openrouter.ChatMessagePartTypeImageURL,
			ImageURL: &openrouter.ChatMessageImageURL{
				URL: data,
			},
		})

	case llm.AttachmentSourceURL:
		parts = append(parts, openrouter.ChatMessagePart{
			Type: openrouter.ChatMessagePartTypeImageURL,
			ImageURL: &openrouter.ChatMessageImageURL{
				URL: attachment.Data(),
			},
		})

	default:
		return openrouter.Content{}, errors.Errorf("unsupported attachment source: %s", attachment.Source())
	}

	return openrouter.Content{
		Multi: parts,
	}, nil
}

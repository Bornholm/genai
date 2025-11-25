package openai

import (
	"fmt"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/pkg/errors"
)

// OpenAIAttachmentValidator implements provider-specific validation for OpenAI
type OpenAIAttachmentValidator struct {
	model string
}

// ValidateAttachment implements llm.AttachmentValidator - Layer 2 validation
func (v *OpenAIAttachmentValidator) ValidateAttachment(attachment llm.Attachment) error {

	// Validate attachment type support
	switch attachment.Type() {
	case llm.AttachmentTypeImage:
		return v.validateImageAttachment(attachment)
	case llm.AttachmentTypeAudio:
		return llm.NewAttachmentError("provider", "type", "audio attachments not yet supported by OpenAI provider")
	case llm.AttachmentTypeVideo:
		return llm.NewAttachmentError("provider", "type", "video attachments not yet supported by OpenAI provider")
	case llm.AttachmentTypeDocument:
		return llm.NewAttachmentError("provider", "type", "document attachments not yet supported by OpenAI provider")
	default:
		return llm.NewAttachmentError("provider", "type", fmt.Sprintf("unsupported attachment type: %s", attachment.Type()))
	}
}

// validateImageAttachment validates image-specific constraints for OpenAI
func (v *OpenAIAttachmentValidator) validateImageAttachment(attachment llm.Attachment) error {
	// Check supported MIME types
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
		return llm.NewAttachmentError("provider", "mime_type", fmt.Sprintf("unsupported image MIME type: %s (supported: %v)", attachment.MimeType(), supportedMimeTypes))
	}

	// For base64 attachments, check size limits (approximate)
	if attachment.Source() == llm.AttachmentSourceBase64 {
		data := attachment.Data()
		// Remove data URL prefix if present
		if strings.HasPrefix(data, "data:") {
			parts := strings.SplitN(data, ",", 2)
			if len(parts) == 2 {
				data = parts[1]
			}
		}

		// Approximate size check (base64 is ~33% larger than binary)
		// OpenAI limit is 20MB for images
		maxBase64Size := 20 * 1024 * 1024 * 4 / 3 // ~26.7MB in base64
		if len(data) > maxBase64Size {
			return llm.NewAttachmentError("provider", "size", fmt.Sprintf("image size exceeds OpenAI limit of 20MB (estimated size: %d bytes)", len(data)*3/4))
		}
	}

	return nil
}

// NewOpenAIAttachmentValidator creates a new OpenAI attachment validator
func NewOpenAIAttachmentValidator(model string) *OpenAIAttachmentValidator {
	return &OpenAIAttachmentValidator{
		model: model,
	}
}

var _ llm.AttachmentValidator = &OpenAIAttachmentValidator{}

// ConvertAttachmentToContentPart converts an attachment to OpenAI content part format
func ConvertAttachmentToContentPart(attachment llm.Attachment) (openai.ChatCompletionContentPartUnionParam, error) {
	switch attachment.Type() {
	case llm.AttachmentTypeImage:
		return convertImageAttachment(attachment)
	default:
		return openai.ChatCompletionContentPartUnionParam{}, errors.Errorf("unsupported attachment type for conversion: %s", attachment.Type())
	}
}

// convertImageAttachment converts an image attachment to OpenAI image content part
func convertImageAttachment(attachment llm.Attachment) (openai.ChatCompletionContentPartUnionParam, error) {
	switch attachment.Source() {
	case llm.AttachmentSourceBase64:
		// For base64, we need to format as data URL if not already
		data := attachment.Data()
		if !strings.HasPrefix(data, "data:") {
			data = fmt.Sprintf("data:%s;base64,%s", attachment.MimeType(), data)
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

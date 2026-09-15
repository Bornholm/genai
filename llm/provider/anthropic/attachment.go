package anthropic

import (
	"encoding/base64"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// attachmentBlock converts an llm.Attachment to a Messages API content block.
//
// Images (png, jpeg, gif, webp) and PDF documents can be sent inline or by
// URL; text/* documents are decoded and sent as plain text sources. Audio
// and video have no equivalent in the Messages API.
func attachmentBlock(attachment llm.Attachment) (anthropicsdk.ContentBlockParamUnion, error) {
	mimeType := strings.ToLower(attachment.MimeType())

	switch attachment.Type() {
	case llm.AttachmentTypeImage:
		if !supportedImageMimeTypes[mimeType] {
			return anthropicsdk.ContentBlockParamUnion{}, errors.Errorf("unsupported image MIME type: %s (supported: image/png, image/jpeg, image/gif, image/webp)", attachment.MimeType())
		}
		switch attachment.Source() {
		case llm.AttachmentSourceBase64:
			data, err := stripDataURL(attachment.Data())
			if err != nil {
				return anthropicsdk.ContentBlockParamUnion{}, err
			}
			return anthropicsdk.ContentBlockParamUnion{OfImage: &anthropicsdk.ImageBlockParam{
				Source: anthropicsdk.ImageBlockParamSourceUnion{OfBase64: &anthropicsdk.Base64ImageSourceParam{
					Data:      data,
					MediaType: anthropicsdk.Base64ImageSourceMediaType(mimeType),
				}},
			}}, nil
		case llm.AttachmentSourceURL:
			return anthropicsdk.ContentBlockParamUnion{OfImage: &anthropicsdk.ImageBlockParam{
				Source: anthropicsdk.ImageBlockParamSourceUnion{OfURL: &anthropicsdk.URLImageSourceParam{
					URL: attachment.Data(),
				}},
			}}, nil
		}

	case llm.AttachmentTypeDocument:
		switch {
		case mimeType == "application/pdf":
			switch attachment.Source() {
			case llm.AttachmentSourceBase64:
				data, err := stripDataURL(attachment.Data())
				if err != nil {
					return anthropicsdk.ContentBlockParamUnion{}, err
				}
				return anthropicsdk.ContentBlockParamUnion{OfDocument: &anthropicsdk.DocumentBlockParam{
					Source: anthropicsdk.DocumentBlockParamSourceUnion{OfBase64: &anthropicsdk.Base64PDFSourceParam{
						Data: data,
					}},
				}}, nil
			case llm.AttachmentSourceURL:
				return anthropicsdk.ContentBlockParamUnion{OfDocument: &anthropicsdk.DocumentBlockParam{
					Source: anthropicsdk.DocumentBlockParamSourceUnion{OfURL: &anthropicsdk.URLPDFSourceParam{
						URL: attachment.Data(),
					}},
				}}, nil
			}

		case strings.HasPrefix(mimeType, "text/"):
			if attachment.Source() != llm.AttachmentSourceBase64 {
				return anthropicsdk.ContentBlockParamUnion{}, errors.New("URL-based text documents are not supported; please download and embed the content as base64")
			}
			payload, err := stripDataURL(attachment.Data())
			if err != nil {
				return anthropicsdk.ContentBlockParamUnion{}, err
			}
			decoded, err := base64.StdEncoding.DecodeString(payload)
			if err != nil {
				return anthropicsdk.ContentBlockParamUnion{}, errors.Wrap(err, "could not decode text document")
			}
			return anthropicsdk.ContentBlockParamUnion{OfDocument: &anthropicsdk.DocumentBlockParam{
				Source: anthropicsdk.DocumentBlockParamSourceUnion{OfText: &anthropicsdk.PlainTextSourceParam{
					Data: string(decoded),
				}},
			}}, nil

		default:
			return anthropicsdk.ContentBlockParamUnion{}, errors.Errorf("unsupported document MIME type: %s (supported: application/pdf, text/*)", attachment.MimeType())
		}
	}

	return anthropicsdk.ContentBlockParamUnion{}, errors.Errorf("unsupported attachment type '%s' (source '%s')", attachment.Type(), attachment.Source())
}

// toolResultBlockContent converts an attachment to the narrower union a
// tool_result block accepts (text, image, document).
func toolResultBlockContent(attachment llm.Attachment) (anthropicsdk.ToolResultBlockParamContentUnion, error) {
	block, err := attachmentBlock(attachment)
	if err != nil {
		return anthropicsdk.ToolResultBlockParamContentUnion{}, err
	}
	switch {
	case block.OfImage != nil:
		return anthropicsdk.ToolResultBlockParamContentUnion{OfImage: block.OfImage}, nil
	case block.OfDocument != nil:
		return anthropicsdk.ToolResultBlockParamContentUnion{OfDocument: block.OfDocument}, nil
	}
	return anthropicsdk.ToolResultBlockParamContentUnion{}, errors.Errorf("unsupported tool result attachment type '%s'", attachment.Type())
}

// supportedImageMimeTypes lists the image formats the Messages API accepts.
var supportedImageMimeTypes = map[string]bool{
	"image/png":  true,
	"image/jpeg": true,
	"image/gif":  true,
	"image/webp": true,
}

// stripDataURL returns the payload of a "data:<mime>;base64,<payload>" URL,
// or the input unchanged when it is already a bare base64 string. A data
// URL without the base64 marker carries a raw payload the API cannot take
// as-is, so it is refused rather than forwarded as corrupt bytes.
func stripDataURL(data string) (string, error) {
	if !strings.HasPrefix(data, "data:") {
		return data, nil
	}
	header, payload, found := strings.Cut(data, ",")
	if !found {
		return "", errors.New("malformed data URL: no payload")
	}
	if !strings.HasSuffix(header, ";base64") {
		return "", errors.New("unsupported data URL: only base64-encoded payloads are accepted")
	}
	return payload, nil
}

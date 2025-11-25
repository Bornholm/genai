package llm

import (
	"encoding/base64"
	"fmt"
	"net/url"
	"regexp"
	"strings"

	"github.com/pkg/errors"
)

// AttachmentType represents the type of attachment
type AttachmentType string

const (
	AttachmentTypeImage    AttachmentType = "image"
	AttachmentTypeAudio    AttachmentType = "audio"
	AttachmentTypeVideo    AttachmentType = "video"
	AttachmentTypeDocument AttachmentType = "document"
)

// AttachmentSource represents how the attachment data is provided
type AttachmentSource string

const (
	AttachmentSourceBase64 AttachmentSource = "base64"
	AttachmentSourceURL    AttachmentSource = "url"
)

// Attachment represents a file attachment with basic format validation
type Attachment interface {
	Type() AttachmentType
	MimeType() string
	Source() AttachmentSource
	Data() string
	ValidateFormat() error
}

// AttachmentValidator provides provider-specific validation
type AttachmentValidator interface {
	ValidateAttachment(attachment Attachment) error
}

// BaseAttachment provides common attachment functionality
type BaseAttachment struct {
	attachmentType AttachmentType
	mimeType       string
	source         AttachmentSource
	data           string
}

// Type implements Attachment
func (a *BaseAttachment) Type() AttachmentType {
	return a.attachmentType
}

// MimeType implements Attachment
func (a *BaseAttachment) MimeType() string {
	return a.mimeType
}

// Source implements Attachment
func (a *BaseAttachment) Source() AttachmentSource {
	return a.source
}

// Data implements Attachment
func (a *BaseAttachment) Data() string {
	return a.data
}

// ValidateFormat implements Attachment - Layer 1 validation
func (a *BaseAttachment) ValidateFormat() error {
	// Validate MIME type format
	if err := validateMimeType(a.mimeType); err != nil {
		return err
	}

	// Validate data based on source
	switch a.source {
	case AttachmentSourceBase64:
		return validateBase64Data(a.data)
	case AttachmentSourceURL:
		return validateURLData(a.data)
	default:
		return NewAttachmentError("format", "source", fmt.Sprintf("unsupported attachment source: %s", a.source))
	}
}

var _ Attachment = &BaseAttachment{}

// AttachmentError represents validation errors with context
type AttachmentError struct {
	Type    string // "format" or "provider"
	Field   string
	Message string
	Cause   error
}

func (e *AttachmentError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s validation error for field '%s': %s (caused by: %v)", e.Type, e.Field, e.Message, e.Cause)
	}
	return fmt.Sprintf("%s validation error for field '%s': %s", e.Type, e.Field, e.Message)
}

func (e *AttachmentError) Unwrap() error {
	return e.Cause
}

// NewAttachmentError creates a new attachment error
func NewAttachmentError(errorType, field, message string) *AttachmentError {
	return &AttachmentError{
		Type:    errorType,
		Field:   field,
		Message: message,
	}
}

// NewAttachmentErrorWithCause creates a new attachment error with a cause
func NewAttachmentErrorWithCause(errorType, field, message string, cause error) *AttachmentError {
	return &AttachmentError{
		Type:    errorType,
		Field:   field,
		Message: message,
		Cause:   cause,
	}
}

// Common validation errors
var (
	ErrInvalidMimeType = NewAttachmentError("format", "mime_type", "invalid MIME type format")
	ErrInvalidBase64   = NewAttachmentError("format", "data", "invalid base64 encoding")
	ErrInvalidURL      = NewAttachmentError("format", "data", "invalid URL format")
	ErrEmptyData       = NewAttachmentError("format", "data", "attachment data cannot be empty")
)

// MIME type validation regex
var mimeTypeRegex = regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9][a-zA-Z0-9\!\#\$\&\-\^\_]*\/[a-zA-Z0-9][a-zA-Z0-9\!\#\$\&\-\^\_\+]*$`)

// validateMimeType validates MIME type format
func validateMimeType(mimeType string) error {
	if mimeType == "" {
		return NewAttachmentError("format", "mime_type", "MIME type cannot be empty")
	}

	if !mimeTypeRegex.MatchString(mimeType) {
		return NewAttachmentErrorWithCause("format", "mime_type", "invalid MIME type format", fmt.Errorf("got: %s", mimeType))
	}

	return nil
}

// validateBase64Data validates base64 encoded data
func validateBase64Data(data string) error {
	if data == "" {
		return ErrEmptyData
	}

	// Handle data URLs (data:mime/type;base64,data)
	if strings.HasPrefix(data, "data:") {
		parts := strings.SplitN(data, ",", 2)
		if len(parts) != 2 {
			return NewAttachmentError("format", "data", "invalid data URL format")
		}
		data = parts[1]
	}

	// Validate base64 encoding
	if _, err := base64.StdEncoding.DecodeString(data); err != nil {
		return NewAttachmentErrorWithCause("format", "data", "invalid base64 encoding", err)
	}

	return nil
}

// validateURLData validates URL format
func validateURLData(data string) error {
	if data == "" {
		return ErrEmptyData
	}

	parsedURL, err := url.Parse(data)
	if err != nil {
		return NewAttachmentErrorWithCause("format", "data", "invalid URL format", err)
	}

	if parsedURL.Scheme == "" {
		return NewAttachmentError("format", "data", "URL must have a scheme (http, https, etc.)")
	}

	if parsedURL.Host == "" {
		return NewAttachmentError("format", "data", "URL must have a host")
	}

	return nil
}

// NewBase64Attachment creates a new base64-encoded attachment
func NewBase64Attachment(attachmentType AttachmentType, mimeType, data string) (*BaseAttachment, error) {
	attachment := &BaseAttachment{
		attachmentType: attachmentType,
		mimeType:       mimeType,
		source:         AttachmentSourceBase64,
		data:           data,
	}

	if err := attachment.ValidateFormat(); err != nil {
		return nil, errors.WithStack(err)
	}

	return attachment, nil
}

// NewURLAttachment creates a new URL-based attachment
func NewURLAttachment(attachmentType AttachmentType, mimeType, url string) (*BaseAttachment, error) {
	attachment := &BaseAttachment{
		attachmentType: attachmentType,
		mimeType:       mimeType,
		source:         AttachmentSourceURL,
		data:           url,
	}

	if err := attachment.ValidateFormat(); err != nil {
		return nil, errors.WithStack(err)
	}

	return attachment, nil
}

// Helper functions for common attachment types

// NewImageAttachment creates a new image attachment
func NewImageAttachment(mimeType, data string, isURL bool) (*BaseAttachment, error) {
	if isURL {
		return NewURLAttachment(AttachmentTypeImage, mimeType, data)
	}
	return NewBase64Attachment(AttachmentTypeImage, mimeType, data)
}

// NewAudioAttachment creates a new audio attachment
func NewAudioAttachment(mimeType, data string, isURL bool) (*BaseAttachment, error) {
	if isURL {
		return NewURLAttachment(AttachmentTypeAudio, mimeType, data)
	}
	return NewBase64Attachment(AttachmentTypeAudio, mimeType, data)
}

// NewVideoAttachment creates a new video attachment
func NewVideoAttachment(mimeType, data string, isURL bool) (*BaseAttachment, error) {
	if isURL {
		return NewURLAttachment(AttachmentTypeVideo, mimeType, data)
	}
	return NewBase64Attachment(AttachmentTypeVideo, mimeType, data)
}

// NewDocumentAttachment creates a new document attachment
func NewDocumentAttachment(mimeType, data string, isURL bool) (*BaseAttachment, error) {
	if isURL {
		return NewURLAttachment(AttachmentTypeDocument, mimeType, data)
	}
	return NewBase64Attachment(AttachmentTypeDocument, mimeType, data)
}

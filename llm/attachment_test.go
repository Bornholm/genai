package llm

import (
	"strings"
	"testing"
)

func TestNewBase64Attachment(t *testing.T) {
	tests := []struct {
		name        string
		attachType  AttachmentType
		mimeType    string
		data        string
		expectError bool
	}{
		{
			name:        "valid image attachment",
			attachType:  AttachmentTypeImage,
			mimeType:    "image/jpeg",
			data:        "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
			expectError: false,
		},
		{
			name:        "valid base64 without data URL prefix",
			attachType:  AttachmentTypeImage,
			mimeType:    "image/png",
			data:        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
			expectError: false,
		},
		{
			name:        "invalid MIME type",
			attachType:  AttachmentTypeImage,
			mimeType:    "invalid-mime",
			data:        "validbase64data",
			expectError: true,
		},
		{
			name:        "invalid base64 data",
			attachType:  AttachmentTypeImage,
			mimeType:    "image/jpeg",
			data:        "invalid-base64!@#$",
			expectError: true,
		},
		{
			name:        "empty data",
			attachType:  AttachmentTypeImage,
			mimeType:    "image/jpeg",
			data:        "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attachment, err := NewBase64Attachment(tt.attachType, tt.mimeType, tt.data)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if attachment.Type() != tt.attachType {
				t.Errorf("expected type %s, got %s", tt.attachType, attachment.Type())
			}

			if attachment.MimeType() != tt.mimeType {
				t.Errorf("expected MIME type %s, got %s", tt.mimeType, attachment.MimeType())
			}

			if attachment.Source() != AttachmentSourceBase64 {
				t.Errorf("expected source %s, got %s", AttachmentSourceBase64, attachment.Source())
			}

			if attachment.Data() != tt.data {
				t.Errorf("expected data %s, got %s", tt.data, attachment.Data())
			}
		})
	}
}

func TestNewURLAttachment(t *testing.T) {
	tests := []struct {
		name        string
		attachType  AttachmentType
		mimeType    string
		url         string
		expectError bool
	}{
		{
			name:        "valid HTTPS URL",
			attachType:  AttachmentTypeImage,
			mimeType:    "image/jpeg",
			url:         "https://example.com/image.jpg",
			expectError: false,
		},
		{
			name:        "valid HTTP URL",
			attachType:  AttachmentTypeAudio,
			mimeType:    "audio/mp3",
			url:         "http://example.com/audio.mp3",
			expectError: false,
		},
		{
			name:        "invalid URL - no scheme",
			attachType:  AttachmentTypeImage,
			mimeType:    "image/jpeg",
			url:         "example.com/image.jpg",
			expectError: true,
		},
		{
			name:        "invalid URL - no host",
			attachType:  AttachmentTypeImage,
			mimeType:    "image/jpeg",
			url:         "https://",
			expectError: true,
		},
		{
			name:        "empty URL",
			attachType:  AttachmentTypeImage,
			mimeType:    "image/jpeg",
			url:         "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attachment, err := NewURLAttachment(tt.attachType, tt.mimeType, tt.url)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if attachment.Type() != tt.attachType {
				t.Errorf("expected type %s, got %s", tt.attachType, attachment.Type())
			}

			if attachment.MimeType() != tt.mimeType {
				t.Errorf("expected MIME type %s, got %s", tt.mimeType, attachment.MimeType())
			}

			if attachment.Source() != AttachmentSourceURL {
				t.Errorf("expected source %s, got %s", AttachmentSourceURL, attachment.Source())
			}

			if attachment.Data() != tt.url {
				t.Errorf("expected URL %s, got %s", tt.url, attachment.Data())
			}
		})
	}
}

func TestMultimodalMessage(t *testing.T) {
	// Create test attachments
	imageAttachment, err := NewBase64Attachment(
		AttachmentTypeImage,
		"image/jpeg",
		"data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
	)
	if err != nil {
		t.Fatalf("failed to create image attachment: %v", err)
	}

	audioAttachment, err := NewURLAttachment(
		AttachmentTypeAudio,
		"audio/mp3",
		"https://example.com/audio.mp3",
	)
	if err != nil {
		t.Fatalf("failed to create audio attachment: %v", err)
	}

	// Test multimodal message creation
	message := NewMultimodalMessage(
		RoleUser,
		"Please analyze these files.",
		imageAttachment,
		audioAttachment,
	)

	if message.Role() != RoleUser {
		t.Errorf("expected role %s, got %s", RoleUser, message.Role())
	}

	if message.Content() != "Please analyze these files." {
		t.Errorf("expected content 'Please analyze these files.', got '%s'", message.Content())
	}

	attachments := message.Attachments()
	if len(attachments) != 2 {
		t.Errorf("expected 2 attachments, got %d", len(attachments))
	}

	if !message.HasAttachments() {
		t.Error("expected HasAttachments() to return true")
	}

	// Verify attachments
	if attachments[0].Type() != AttachmentTypeImage {
		t.Errorf("expected first attachment to be image, got %s", attachments[0].Type())
	}

	if attachments[1].Type() != AttachmentTypeAudio {
		t.Errorf("expected second attachment to be audio, got %s", attachments[1].Type())
	}
}

func TestAttachmentError(t *testing.T) {
	err := NewAttachmentError("format", "mime_type", "invalid MIME type")

	expectedMsg := "format validation error for field 'mime_type': invalid MIME type"
	if err.Error() != expectedMsg {
		t.Errorf("expected error message '%s', got '%s'", expectedMsg, err.Error())
	}

	if err.Type != "format" {
		t.Errorf("expected error type 'format', got '%s'", err.Type)
	}

	if err.Field != "mime_type" {
		t.Errorf("expected error field 'mime_type', got '%s'", err.Field)
	}
}

func TestValidateMimeType(t *testing.T) {
	tests := []struct {
		mimeType    string
		expectError bool
	}{
		{"image/jpeg", false},
		{"image/png", false},
		{"audio/mp3", false},
		{"video/mp4", false},
		{"application/pdf", false},
		{"text/plain", false},
		{"", true},
		{"invalid", true},
		{"image/", true},
		{"/jpeg", true},
		{"image//jpeg", true},
	}

	for _, tt := range tests {
		t.Run(tt.mimeType, func(t *testing.T) {
			err := validateMimeType(tt.mimeType)

			if tt.expectError && err == nil {
				t.Errorf("expected error for MIME type '%s' but got none", tt.mimeType)
			}

			if !tt.expectError && err != nil {
				t.Errorf("unexpected error for MIME type '%s': %v", tt.mimeType, err)
			}
		})
	}
}

func TestChatCompletionOptionsValidation(t *testing.T) {
	// Test validation with attachments
	imageAttachment, err := NewBase64Attachment(
		AttachmentTypeImage,
		"image/jpeg",
		"data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
	)
	if err != nil {
		t.Fatalf("failed to create image attachment: %v", err)
	}

	// Test message with only attachments (no text content)
	messageWithOnlyAttachments := NewMultimodalMessage(
		RoleUser,
		"", // empty content
		imageAttachment,
	)

	opts := &ChatCompletionOptions{
		Messages:    []Message{messageWithOnlyAttachments},
		Temperature: 0.7,
	}

	err = opts.Validate()
	if err != nil {
		t.Errorf("validation should pass for message with attachments but no text: %v", err)
	}

	// Test message with invalid attachment
	invalidAttachment := &BaseAttachment{
		attachmentType: AttachmentTypeImage,
		mimeType:       "invalid-mime",
		source:         AttachmentSourceBase64,
		data:           "invalid-base64",
	}

	messageWithInvalidAttachment := NewMultimodalMessage(
		RoleUser,
		"Test message",
		invalidAttachment,
	)

	opts.Messages = []Message{messageWithInvalidAttachment}
	err = opts.Validate()
	if err == nil {
		t.Error("validation should fail for message with invalid attachment")
	}

	if !strings.Contains(err.Error(), "validation failed") {
		t.Errorf("error should mention validation failure, got: %v", err)
	}
}

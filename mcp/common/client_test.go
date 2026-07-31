package common

import (
	"context"
	"encoding/base64"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
	goMCP "github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestToToolResultKeepsMedia guards the conversion of a tool call result: a
// server answering with an image (an MCP server exposing a "fetch_image" tool,
// for instance) must not look like it answered with nothing.
func TestToToolResultKeepsMedia(t *testing.T) {
	imageData := []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}

	res := &goMCP.CallToolResult{
		Content: []goMCP.Content{
			&goMCP.TextContent{Text: "Here is the screen."},
			&goMCP.ImageContent{MIMEType: "image/png", Data: imageData},
		},
	}

	result := toToolResult(context.Background(), res)

	attachments := result.Attachments()
	if len(attachments) != 1 {
		t.Fatalf("attachments: expected 1, got %d", len(attachments))
	}

	if e, g := llm.AttachmentTypeImage, attachments[0].Type(); e != g {
		t.Errorf("attachment type: expected %q, got %q", e, g)
	}

	if e, g := "image/png", attachments[0].MimeType(); e != g {
		t.Errorf("attachment mime type: expected %q, got %q", e, g)
	}

	// The SDK hands over the decoded bytes; the attachment layer expects them
	// base64-encoded.
	if e, g := base64.StdEncoding.EncodeToString(imageData), attachments[0].Data(); e != g {
		t.Errorf("attachment data: expected %q, got %q", e, g)
	}

	if !strings.Contains(result.Text(), "Here is the screen.") {
		t.Errorf("text: expected the text content, got %q", result.Text())
	}

	// The marker is what keeps a caller unable to forward attachments honest:
	// without it the model would answer that no image came back.
	if !strings.Contains(result.Text(), "image/png") {
		t.Errorf("text: expected a marker naming the medium, got %q", result.Text())
	}
}

func TestToToolResultEmbeddedResource(t *testing.T) {
	blob := []byte("%PDF-1.4")

	res := &goMCP.CallToolResult{
		Content: []goMCP.Content{
			&goMCP.EmbeddedResource{Resource: &goMCP.ResourceContents{URI: "file:///note.md", Text: "some notes"}},
			&goMCP.EmbeddedResource{Resource: &goMCP.ResourceContents{URI: "file:///doc.pdf", MIMEType: "application/pdf", Blob: blob}},
			&goMCP.ResourceLink{URI: "amoxtli://images/abc123"},
		},
	}

	result := toToolResult(context.Background(), res)

	if !strings.Contains(result.Text(), "some notes") {
		t.Errorf("text: expected the textual resource, got %q", result.Text())
	}

	if !strings.Contains(result.Text(), "amoxtli://images/abc123") {
		t.Errorf("text: expected the resource link URI, got %q", result.Text())
	}

	attachments := result.Attachments()
	if len(attachments) != 1 {
		t.Fatalf("attachments: expected 1, got %d", len(attachments))
	}

	if e, g := llm.AttachmentTypeDocument, attachments[0].Type(); e != g {
		t.Errorf("attachment type: expected %q, got %q", e, g)
	}
}

// An unusable medium costs the attachment, never the call: the text of the
// result may well be the part that matters.
func TestToToolResultIgnoresInvalidAttachment(t *testing.T) {
	res := &goMCP.CallToolResult{
		Content: []goMCP.Content{
			&goMCP.TextContent{Text: "partial answer"},
			&goMCP.ImageContent{MIMEType: "not-a-mime-type", Data: []byte{0x01}},
		},
	}

	result := toToolResult(context.Background(), res)

	if len(result.Attachments()) != 0 {
		t.Errorf("attachments: expected none, got %d", len(result.Attachments()))
	}

	if e, g := "partial answer", result.Text(); e != g {
		t.Errorf("text: expected %q, got %q", e, g)
	}
}

func TestToToolResultError(t *testing.T) {
	res := &goMCP.CallToolResult{
		IsError: true,
		Content: []goMCP.Content{&goMCP.TextContent{Text: "boom"}},
	}

	result := toToolResult(context.Background(), res)

	if e, g := "ERROR:\nboom", result.Text(); e != g {
		t.Errorf("text: expected %q, got %q", e, g)
	}
}

func TestAttachmentTypeOf(t *testing.T) {
	testCases := []struct {
		MimeType string
		Expected llm.AttachmentType
	}{
		{"image/png", llm.AttachmentTypeImage},
		{"audio/mpeg", llm.AttachmentTypeAudio},
		{"video/mp4", llm.AttachmentTypeVideo},
		{"application/pdf", llm.AttachmentTypeDocument},
		{"", llm.AttachmentTypeDocument},
	}

	for _, tc := range testCases {
		if e, g := tc.Expected, attachmentTypeOf(tc.MimeType); e != g {
			t.Errorf("attachmentTypeOf(%q): expected %q, got %q", tc.MimeType, e, g)
		}
	}
}

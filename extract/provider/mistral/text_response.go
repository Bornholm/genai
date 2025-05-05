package mistral

import (
	"bytes"
	"io"

	"github.com/bornholm/genai/extract"
)

type TextResponse struct {
	format extract.TextFormat
	output *bytes.Buffer
}

// Format implements extract.TextResponse.
func (r *TextResponse) Format() extract.TextFormat {
	return r.format
}

// Output implements extract.TextResponse.
func (r *TextResponse) Output() io.Reader {
	return r.output
}

var _ extract.TextResponse = &TextResponse{}

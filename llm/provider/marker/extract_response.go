package marker

import (
	"bytes"
	"io"

	"github.com/bornholm/genai/llm"
)

type ExtractTextResponse struct {
	format   llm.ExtractTextFormat
	output   *bytes.Buffer
	Images   map[string]string
	Metadata any
}

// Format implements llm.ExtractTextResponse.
func (r *ExtractTextResponse) Format() llm.ExtractTextFormat {
	return r.format
}

// Output implements llm.ExtractTextResponse.
func (r *ExtractTextResponse) Output() io.Reader {
	return r.output
}

var _ llm.ExtractTextResponse = &ExtractTextResponse{}

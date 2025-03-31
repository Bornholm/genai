package mistral

import (
	"context"

	"github.com/bornholm/genai/llm"
)

type OCRClient struct {
}

// OCR implements llm.OCRClient.
func (c *OCRClient) OCR(ctx context.Context, funcs ...llm.OCROptionFunc) (llm.OCRResponse, error) {
	panic("unimplemented")
}

func NewOCRClient() *OCRClient {
	return &OCRClient{}
}

var _ llm.OCRClient = &OCRClient{}

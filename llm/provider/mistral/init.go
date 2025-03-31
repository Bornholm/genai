package mistral

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

const Name provider.Name = "mistral"

func init() {
	provider.RegisterOCR(Name, func(ctx context.Context, opts provider.ClientOptions) (llm.OCRClient, error) {
		return NewOCRClient(), nil
	})
}

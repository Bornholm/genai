package mistral

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

const Name provider.Name = "mistral"

func init() {
	provider.Register(Name, func(ctx context.Context) (llm.Client, error) {
		return NewClient(), nil
	})
}

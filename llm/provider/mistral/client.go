package mistral

import (
	"context"

	"github.com/bornholm/genai/llm"
)

type Client struct {
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.CompletionResponse, error) {
	panic("unimplemented")
}

func NewClient() *Client {
	return &Client{}
}

var _ llm.Client = &Client{}

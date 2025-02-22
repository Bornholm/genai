package mistral

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type Client struct {
}

// Model implements llm.Client.
func (c *Client) Model() string {
	return ""
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.CompletionResponse, error) {
	return nil, errors.WithStack(llm.ErrNotImplemented)
}

func NewClient() *Client {
	return &Client{}
}

var _ llm.Client = &Client{}

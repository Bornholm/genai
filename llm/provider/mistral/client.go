package mistral

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type Client struct {
}

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, errors.WithStack(llm.ErrUnavailable)
}

// Model implements llm.Client.
func (c *Client) Model() string {
	return ""
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.CompletionResponse, error) {
	return nil, errors.WithStack(llm.ErrUnavailable)
}

func NewClient() *Client {
	return &Client{}
}

var _ llm.Client = &Client{}

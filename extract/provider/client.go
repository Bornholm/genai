package provider

import (
	"context"

	"github.com/bornholm/genai/extract"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type Client struct {
	textClient extract.TextClient
}

// Text implements extract.Client.
func (c *Client) Text(ctx context.Context, funcs ...extract.TextOptionFunc) (extract.TextResponse, error) {
	if c.textClient == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	response, err := c.textClient.Text(ctx, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return response, nil
}

func NewClient(textClient extract.TextClient) *Client {
	return &Client{
		textClient: textClient,
	}
}

var _ extract.Client = &Client{}

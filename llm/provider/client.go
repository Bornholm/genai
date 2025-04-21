package provider

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type Client struct {
	chatCompletion llm.ChatCompletionClient
	embeddings     llm.EmbeddingsClient
	extractText    llm.ExtractTextClient
}

// ExtractText implements llm.Client.
func (c *Client) ExtractText(ctx context.Context, funcs ...llm.ExtractTextOptionFunc) (llm.ExtractTextResponse, error) {
	if c.extractText == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	response, err := c.extractText.ExtractText(ctx, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return response, nil
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	if c.chatCompletion == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	response, err := c.chatCompletion.ChatCompletion(ctx, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return response, nil
}

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, input string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	if c.embeddings == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	response, err := c.embeddings.Embeddings(ctx, input, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return response, nil
}

func NewClient(chatCompletion llm.ChatCompletionClient, embeddings llm.EmbeddingsClient, extractText llm.ExtractTextClient) *Client {
	return &Client{
		chatCompletion: chatCompletion,
		embeddings:     embeddings,
		extractText:    extractText,
	}
}

var _ llm.Client = &Client{}

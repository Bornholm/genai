package ratelimit

import (
	"context"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"golang.org/x/time/rate"
)

type Client struct {
	limiter *rate.Limiter
	client  llm.Client
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	if err := c.limiter.Wait(ctx); err != nil {
		return nil, errors.WithStack(err)
	}
	return c.client.ChatCompletion(ctx, funcs...)
}

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, input string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	if err := c.limiter.Wait(ctx); err != nil {
		return nil, errors.WithStack(err)
	}
	return c.client.Embeddings(ctx, input, funcs...)
}

func Wrap(client llm.Client, minInterval time.Duration, maxBurst int) *Client {
	return &Client{
		limiter: rate.NewLimiter(rate.Every(minInterval), maxBurst),
		client:  client,
	}
}

var _ llm.Client = &Client{}

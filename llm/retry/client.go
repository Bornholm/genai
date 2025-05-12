package retry

import (
	"context"
	"log/slog"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type Client struct {
	baseDelay  time.Duration
	maxRetries int
	client     llm.Client
}

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, input string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	backoff := c.baseDelay
	maxRetries := c.maxRetries
	retries := 0

	for {
		res, err := c.client.Embeddings(ctx, input, funcs...)
		if err != nil {
			if retries >= maxRetries {
				return nil, errors.WithStack(err)
			}

			if errors.Is(err, llm.ErrRateLimit) {
				slog.DebugContext(ctx, "request failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", errors.WithStack(err)))

				retries++
				time.Sleep(backoff)
				backoff *= 2
				continue
			}

			return nil, errors.WithStack(err)
		}

		return res, nil
	}
}

// ChatCompletion implements llm.ChatCompletionClient.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	backoff := c.baseDelay
	maxRetries := c.maxRetries
	retries := 0

	for {
		res, err := c.client.ChatCompletion(ctx, funcs...)
		if err != nil {
			if retries >= maxRetries {
				return nil, errors.WithStack(err)
			}

			if errors.Is(err, llm.ErrRateLimit) {
				slog.DebugContext(ctx, "request failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", errors.WithStack(err)))

				retries++
				time.Sleep(backoff)
				backoff *= 2
				continue
			}

			return nil, errors.WithStack(err)
		}

		return res, nil
	}
}

func Wrap(client llm.Client, baseDelay time.Duration, maxRetries int) *Client {
	return &Client{
		baseDelay:  baseDelay,
		maxRetries: maxRetries,
		client:     client,
	}
}

var _ llm.Client = &Client{}

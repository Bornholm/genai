package retry

import (
	"context"
	stderrors "errors"
	"log/slog"
	"time"

	"github.com/bornholm/genai/extract"
	"github.com/pkg/errors"
)

type Client struct {
	baseDelay  time.Duration
	maxRetries int
	client     extract.Client
}

// Text implements [extract.Client].
func (c *Client) Text(ctx context.Context, funcs ...extract.TextOptionFunc) (extract.TextResponse, error) {
	backoff := c.baseDelay
	maxRetries := c.maxRetries
	retries := 0

	for {
		res, err := c.client.Text(ctx, funcs...)
		if err != nil {
			if retries >= maxRetries {
				return nil, errors.WithStack(err)
			}

			if errors.Is(err, extract.ErrRateLimit) {
				slog.DebugContext(ctx, "request failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", errors.WithStack(err)))

				retries++
				if !wait(ctx, backoff) {
					// Both stay visible: the caller left, and why it was waiting.
					return nil, errors.WithStack(stderrors.Join(ctx.Err(), err))
				}
				backoff *= 2
				continue
			}

			return nil, errors.WithStack(err)
		}

		return res, nil
	}
}

func NewClient(client extract.Client, baseDelay time.Duration, maxRetries int) *Client {
	return &Client{
		baseDelay:  baseDelay,
		maxRetries: maxRetries,
		client:     client,
	}
}

var _ extract.Client = &Client{}

// wait waits for d, or until ctx ends, and reports whether d elapsed.
func wait(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}

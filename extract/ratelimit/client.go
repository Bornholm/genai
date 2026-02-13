package ratelimit

import (
	"context"
	"time"

	"github.com/bornholm/genai/extract"
	"github.com/pkg/errors"
	"golang.org/x/time/rate"
)

type Client struct {
	limiter *rate.Limiter
	client  extract.Client
}

// Text implements [extract.Client].
func (c *Client) Text(ctx context.Context, funcs ...extract.TextOptionFunc) (extract.TextResponse, error) {
	if err := c.limiter.Wait(ctx); err != nil {
		return nil, errors.WithStack(err)
	}

	return c.client.Text(ctx, funcs...)
}

func NewClient(client extract.Client, interval time.Duration, maxBurst int) *Client {
	return &Client{
		limiter: rate.NewLimiter(rate.Every(interval), maxBurst),
		client:  client,
	}
}

var _ extract.Client = &Client{}

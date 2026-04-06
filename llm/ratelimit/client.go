package ratelimit

import (
	"context"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"golang.org/x/time/rate"
)

type Client struct {
	chatLimiter       *rate.Limiter
	embeddingsLimiter *rate.Limiter
	client            llm.Client
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	if err := c.chatLimiter.Wait(ctx); err != nil {
		return nil, errors.WithStack(err)
	}
	return c.client.ChatCompletion(ctx, funcs...)
}

// ChatCompletionStream implements llm.Client.
func (c *Client) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	if err := c.chatLimiter.Wait(ctx); err != nil {
		return nil, errors.WithStack(err)
	}
	return c.client.ChatCompletionStream(ctx, funcs...)
}

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	if err := c.embeddingsLimiter.Wait(ctx); err != nil {
		return nil, errors.WithStack(err)
	}
	return c.client.Embeddings(ctx, inputs, funcs...)
}

type Options struct {
	ChatMinInterval       time.Duration
	ChatMaxBurst          int
	EmbeddingsMinInterval time.Duration
	EmbeddingsMaxBurst    int
}

type OptionFunc func(*Options)

func WithChatLimit(minInterval time.Duration, maxBurst int) OptionFunc {
	return func(o *Options) {
		o.ChatMinInterval = minInterval
		o.ChatMaxBurst = maxBurst
	}
}

func WithEmbeddingsLimit(minInterval time.Duration, maxBurst int) OptionFunc {
	return func(o *Options) {
		o.EmbeddingsMinInterval = minInterval
		o.EmbeddingsMaxBurst = maxBurst
	}
}

func NewClient(client llm.Client, funcs ...OptionFunc) *Client {
	opts := &Options{
		ChatMinInterval:       time.Second,
		ChatMaxBurst:          1,
		EmbeddingsMinInterval: time.Second,
		EmbeddingsMaxBurst:    1,
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return &Client{
		chatLimiter:       rate.NewLimiter(rate.Every(opts.ChatMinInterval), opts.ChatMaxBurst),
		embeddingsLimiter: rate.NewLimiter(rate.Every(opts.EmbeddingsMinInterval), opts.EmbeddingsMaxBurst),
		client:            client,
	}
}

var _ llm.Client = &Client{}

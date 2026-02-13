package tokenlimit

import (
	"time"

	"golang.org/x/time/rate"
)

type Options struct {
	ChatCompletionLimiter *rate.Limiter
	EmbeddingsLimiter     *rate.Limiter
}

type OptionFunc func(opts *Options)

func NewOptions(funcs ...OptionFunc) *Options {
	opts := &Options{
		ChatCompletionLimiter: rate.NewLimiter(rate.Every(time.Minute), 500000),
		EmbeddingsLimiter:     rate.NewLimiter(rate.Every(time.Minute), 20000000),
	}

	for _, fn := range funcs {
		fn(opts)
	}

	return opts
}

func WithChatCompletionLimit(max int, interval time.Duration) OptionFunc {
	return func(opts *Options) {
		opts.ChatCompletionLimiter = rate.NewLimiter(rate.Every(interval), max)
	}
}

func WithEmbeddingsLimit(max int, interval time.Duration) OptionFunc {
	return func(opts *Options) {
		opts.EmbeddingsLimiter = rate.NewLimiter(rate.Every(interval), max)
	}
}

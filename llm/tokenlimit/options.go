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
		// rate.Limit(float64(max)/interval.Seconds()) means "max tokens per interval",
		// NOT rate.Every(interval) which means "1 token per interval".
		ChatCompletionLimiter: rate.NewLimiter(rate.Limit(float64(500000)/time.Minute.Seconds()), 500000),
		EmbeddingsLimiter:     rate.NewLimiter(rate.Limit(float64(20000000)/time.Minute.Seconds()), 20000000),
	}

	for _, fn := range funcs {
		fn(opts)
	}

	return opts
}

func WithChatCompletionLimit(max int, interval time.Duration) OptionFunc {
	return func(opts *Options) {
		opts.ChatCompletionLimiter = rate.NewLimiter(rate.Limit(float64(max)/interval.Seconds()), max)
	}
}

func WithEmbeddingsLimit(max int, interval time.Duration) OptionFunc {
	return func(opts *Options) {
		opts.EmbeddingsLimiter = rate.NewLimiter(rate.Limit(float64(max)/interval.Seconds()), max)
	}
}

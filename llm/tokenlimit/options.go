package tokenlimit

import (
	"time"

	"golang.org/x/time/rate"
)

// DefaultDrainTimeout caps how long an abandoned upstream stream is drained
// for. A client that honours its context closes its channel at once; this only
// bounds what one that honours nothing can hold.
const DefaultDrainTimeout = 30 * time.Second

type Options struct {
	ChatCompletionLimiter *rate.Limiter
	EmbeddingsLimiter     *rate.Limiter
	TranscriptionLimiter  *rate.Limiter
	// DrainTimeout caps how long an abandoned upstream stream is drained for.
	// Zero or less means no limit. Default DefaultDrainTimeout.
	DrainTimeout time.Duration
}

type OptionFunc func(opts *Options)

func NewOptions(funcs ...OptionFunc) *Options {
	opts := &Options{
		// rate.Limit(float64(max)/interval.Seconds()) means "max tokens per interval",
		// NOT rate.Every(interval) which means "1 token per interval".
		ChatCompletionLimiter: rate.NewLimiter(rate.Limit(float64(500000)/time.Minute.Seconds()), 500000),
		EmbeddingsLimiter:     rate.NewLimiter(rate.Limit(float64(20000000)/time.Minute.Seconds()), 20000000),
		DrainTimeout:          DefaultDrainTimeout,
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

func WithTranscriptionLimit(max int, interval time.Duration) OptionFunc {
	return func(opts *Options) {
		opts.TranscriptionLimiter = rate.NewLimiter(rate.Limit(float64(max)/interval.Seconds()), max)
	}
}

// WithDrainTimeout sets how long an abandoned upstream stream is drained for.
// Zero or less means no limit, not an immediate give-up.
func WithDrainTimeout(timeout time.Duration) OptionFunc {
	return func(opts *Options) {
		opts.DrainTimeout = timeout
	}
}

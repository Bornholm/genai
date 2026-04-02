package stdio

import (
	"os"
	"time"
)

type Options struct {
	Env              []string
	MaxRetries       int
	BaseDelay        time.Duration
	ReconnectEnabled bool
}

type OptionFunc func(opts *Options)

func NewOptions(funcs ...OptionFunc) *Options {
	opts := &Options{
		Env:              os.Environ(),
		MaxRetries:       3,
		BaseDelay:        100 * time.Millisecond,
		ReconnectEnabled: true,
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

func WithEnv(env ...string) OptionFunc {
	return func(opts *Options) {
		opts.Env = env
	}
}

func WithMaxRetries(maxRetries int) OptionFunc {
	return func(opts *Options) {
		opts.MaxRetries = maxRetries
	}
}

func WithBaseDelay(delay time.Duration) OptionFunc {
	return func(opts *Options) {
		opts.BaseDelay = delay
	}
}

func WithReconnectEnabled(enabled bool) OptionFunc {
	return func(opts *Options) {
		opts.ReconnectEnabled = enabled
	}
}

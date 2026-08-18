package streamable

import (
	"net/http"
	"time"
)

type Options struct {
	HTTPClient       *http.Client
	MaxRetries       int
	BaseDelay        time.Duration
	ReconnectEnabled bool
	ReadOnlyHint     bool
	// DisableStandaloneSSE skips the GET stream used for server-initiated
	// messages. Some servers only implement the request/response half of the
	// transport and reject that GET.
	DisableStandaloneSSE bool
}

type OptionFunc func(opts *Options)

func NewOptions(funcs ...OptionFunc) *Options {
	opts := &Options{
		HTTPClient:       http.DefaultClient,
		MaxRetries:       3,
		BaseDelay:        100 * time.Millisecond,
		ReconnectEnabled: true,
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

func WithHTTPClient(client *http.Client) OptionFunc {
	return func(opts *Options) {
		opts.HTTPClient = client
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

func WithReadOnlyHint(enabled bool) OptionFunc {
	return func(opts *Options) {
		opts.ReadOnlyHint = enabled
	}
}

func WithDisableStandaloneSSE(disabled bool) OptionFunc {
	return func(opts *Options) {
		opts.DisableStandaloneSSE = disabled
	}
}

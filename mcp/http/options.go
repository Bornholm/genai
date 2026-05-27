package http

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

func WithAuthToken(authToken string) OptionFunc {
	return func(opts *Options) {
		httpClient := opts.HTTPClient
		if httpClient == nil {
			httpClient = http.DefaultClient
		}

		originalTransport := httpClient.Transport
		if originalTransport == nil {
			originalTransport = http.DefaultTransport
		}

		httpClient.Transport = &injectHeaderTransport{
			transport: originalTransport,
			headers: http.Header{
				"Authorization": []string{"Bearer " + authToken},
			},
		}
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

type injectHeaderTransport struct {
	transport http.RoundTripper
	headers   http.Header
}

// RoundTrip implements [http.RoundTripper].
func (t *injectHeaderTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	for key, values := range t.headers {
		for _, v := range values {
			r.Header.Add(key, v)
		}
	}

	return t.transport.RoundTrip(r)
}

var _ http.RoundTripper = &injectHeaderTransport{}

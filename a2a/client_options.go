package a2a

import "net/http"

// ClientOptions contains configuration for the A2A client
type ClientOptions struct {
	HTTPClient *http.Client
}

// ClientOptionFunc is a function that configures the ClientOptions
type ClientOptionFunc func(*ClientOptions)

// NewClientOptions creates a new ClientOptions with defaults
func NewClientOptions(funcs ...ClientOptionFunc) *ClientOptions {
	opts := &ClientOptions{
		HTTPClient: http.DefaultClient,
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

// WithHTTPClient sets the HTTP client
func WithHTTPClient(client *http.Client) ClientOptionFunc {
	return func(o *ClientOptions) {
		o.HTTPClient = client
	}
}

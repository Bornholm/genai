package proxy

import (
	"time"

	"github.com/bornholm/genai/llm"
)

// Options holds all Server configuration.
type Options struct {
	Addr          string        // listen address, default ":8080"
	Hooks         []Hook        // hooks registered on the server
	DefaultClient llm.Client    // fallback client if no resolver matches
	AuthExtractor AuthExtractor // extracts UserID from requests

	// PostResponseTimeout bounds the post-response hooks of a streamed
	// response. Their context is detached from the request — on a client
	// hangup the request context is already canceled, and a usage hook handed
	// it would lose the accounting — so this budget is what replaces the
	// request's own cancellation. Default 30s.
	PostResponseTimeout time.Duration
	// DrainTimeout bounds how long an abandoned upstream stream is drained
	// for. A provider that honours its context closes its channel right away;
	// this only caps what a client that honours nothing can hold. Default 30s.
	DrainTimeout time.Duration
}

// OptionFunc is a functional option for the Server.
type OptionFunc func(*Options)

// WithAddr sets the listening address.
func WithAddr(addr string) OptionFunc {
	return func(o *Options) {
		o.Addr = addr
	}
}

// WithHook adds a hook to the server.
func WithHook(hook Hook) OptionFunc {
	return func(o *Options) {
		o.Hooks = append(o.Hooks, hook)
	}
}

// WithDefaultClient sets the fallback llm.Client used when no resolver matches.
func WithDefaultClient(client llm.Client) OptionFunc {
	return func(o *Options) {
		o.DefaultClient = client
	}
}

// WithAuthExtractor sets the function used to extract UserID from requests.
func WithAuthExtractor(extractor AuthExtractor) OptionFunc {
	return func(o *Options) {
		o.AuthExtractor = extractor
	}
}

// WithPostResponseTimeout sets how long the post-response hooks of a streamed
// response may take before their context is canceled.
func WithPostResponseTimeout(timeout time.Duration) OptionFunc {
	return func(o *Options) {
		o.PostResponseTimeout = timeout
	}
}

// WithDrainTimeout sets how long an abandoned upstream stream is drained for.
func WithDrainTimeout(timeout time.Duration) OptionFunc {
	return func(o *Options) {
		o.DrainTimeout = timeout
	}
}

func defaultOptions() *Options {
	return &Options{
		Addr:                ":8080",
		AuthExtractor:       BearerTokenExtractor(),
		PostResponseTimeout: 30 * time.Second,
		DrainTimeout:        30 * time.Second,
	}
}

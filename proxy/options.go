package proxy

import "github.com/bornholm/genai/llm"

// Options holds all Server configuration.
type Options struct {
	Addr          string        // listen address, default ":8080"
	Hooks         []Hook        // hooks registered on the server
	DefaultClient llm.Client    // fallback client if no resolver matches
	AuthExtractor AuthExtractor // extracts UserID from requests
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

func defaultOptions() *Options {
	return &Options{
		Addr:          ":8080",
		AuthExtractor: BearerTokenExtractor(),
	}
}

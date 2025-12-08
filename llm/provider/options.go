package provider

import (
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type Options struct {
	ChatCompletion *ClientOptions `envPrefix:"CHAT_COMPLETION_"`
	Embeddings     *ClientOptions `envPrefix:"EMBEDDINGS_"`
}

type ClientOptions struct {
	Provider Name   `env:"PROVIDER"`
	BaseURL  string `env:"BASE_URL"`
	APIKey   string `env:"API_KEY"`
	Model    string `env:"MODEL"`
}

// Validate checks if the ClientOptions are valid
func (opts *ClientOptions) Validate() error {
	if opts.Provider == "" {
		return llm.NewValidationError("provider", "provider is required")
	}
	if opts.Model == "" {
		return llm.NewValidationError("model", "model is required")
	}
	if opts.BaseURL == "" {
		return llm.NewValidationError("base_url", "base URL is required")
	}
	return nil
}

type OptionFunc func(opts *Options) error

func WithChatCompletionOptions(clientOpts ClientOptions) OptionFunc {
	return func(opts *Options) error {
		opts.ChatCompletion = &clientOpts
		return nil
	}
}

func WithEmbeddingsOptions(clientOpts ClientOptions) OptionFunc {
	return func(opts *Options) error {
		opts.Embeddings = &clientOpts
		return nil
	}
}

func NewOptions(funcs ...OptionFunc) (*Options, error) {
	opts := &Options{}
	for _, fn := range funcs {
		if err := fn(opts); err != nil {
			return nil, errors.WithStack(err)
		}
	}
	return opts, nil
}

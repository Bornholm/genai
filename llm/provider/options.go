package provider

import (
	"github.com/caarlos0/env/v11"
	"github.com/joho/godotenv"
	"github.com/pkg/errors"
)

type Options struct {
	ChatCompletion *ClientOptions `envPrefix:"CHAT_COMPLETION_"`
	Embeddings     *ClientOptions `envPrefix:"EMBEDDINGS_"`
	ExtractText    *ClientOptions `envPrefix:"EXTRACT_TEXT_"`
}

type ClientOptions struct {
	Provider Name   `env:"PROVIDER"`
	BaseURL  string `env:"BASE_URL"`
	APIKey   string `env:"API_KEY"`
	Model    string `env:"MODEL"`
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

func WithExtractTextOptions(clientOpts ClientOptions) OptionFunc {
	return func(opts *Options) error {
		opts.ExtractText = &clientOpts
		return nil
	}
}

func WithEnv(variableNamePrefix string, envFiles ...string) OptionFunc {
	return func(opts *Options) error {
		if len(envFiles) > 0 {
			if err := godotenv.Load(envFiles...); err != nil {
				return errors.WithStack(err)
			}
		}

		if err := env.ParseWithOptions(opts, env.Options{Prefix: variableNamePrefix}); err != nil {
			return errors.WithStack(err)
		}

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

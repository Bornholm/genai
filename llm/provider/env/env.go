package env

import (
	"os"

	"github.com/bornholm/genai/llm/provider"
	"github.com/caarlos0/env/v11"
	"github.com/joho/godotenv"
	"github.com/pkg/errors"
)

func With(variableNamePrefix string, envFiles ...string) provider.OptionFunc {
	return func(opts *provider.Options) error {
		if len(envFiles) > 0 {
			if err := godotenv.Load(envFiles...); err != nil && !errors.Is(err, os.ErrNotExist) {
				return errors.WithStack(err)
			}
		}

		nilChatCompletion := opts.ChatCompletion == nil
		if nilChatCompletion {
			opts.ChatCompletion = &provider.ClientOptions{}
		}

		nilEmbeddings := opts.Embeddings == nil
		if nilEmbeddings {
			opts.Embeddings = &provider.ClientOptions{}
		}

		if err := env.ParseWithOptions(opts, env.Options{
			Prefix: variableNamePrefix,
		}); err != nil {
			return errors.WithStack(err)
		}

		if err := opts.ChatCompletion.Validate(); nilChatCompletion && err != nil {
			opts.ChatCompletion = nil
		}

		if err := opts.Embeddings.Validate(); nilEmbeddings && err != nil {
			opts.Embeddings = nil
		}

		return nil
	}
}

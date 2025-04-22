package env

import (
	"github.com/bornholm/genai/llm/provider"
	"github.com/caarlos0/env/v11"
	"github.com/joho/godotenv"
	"github.com/pkg/errors"
)

func With(variableNamePrefix string, envFiles ...string) provider.OptionFunc {
	return func(opts *provider.Options) error {
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

package env

import (
	"os"
	"strings"

	"github.com/bornholm/genai/llm/provider"
	"github.com/caarlos0/env/v11"
	"github.com/joho/godotenv"
	"github.com/pkg/errors"
)

// With retourne une provider.OptionFunc qui peuple les options depuis les variables d'environnement.
// Le parsing s'effectue en deux passes :
//  1. Identification du provider via {prefix}CHAT_COMPLETION_PROVIDER (ou EMBEDDINGS_PROVIDER)
//  2. Peuplement des options spécifiques au provider via {prefix}{TYPE}_{PROVIDER_UPPER}_*
//
// Si le provider n'est pas enregistré, Specific reste nil et l'erreur sera levée à Create().
// Les options spécifiques sont initialisées avec les valeurs par défaut du provider (newOptions())
// avant d'être écrasées par les variables d'environnement.
func With(variableNamePrefix string, envFiles ...string) provider.OptionFunc {
	return func(opts *provider.Options) error {
		if len(envFiles) > 0 {
			if err := godotenv.Load(envFiles...); err != nil && !errors.Is(err, os.ErrNotExist) {
				return errors.WithStack(err)
			}
		}

		// Chat completion
		chatResolved, err := resolveOptions(
			variableNamePrefix+"CHAT_COMPLETION_",
			provider.NewChatCompletionProviderOptions,
		)
		if err != nil {
			return errors.Wrap(err, "could not resolve chat completion options")
		}
		opts.ChatCompletion = chatResolved

		// Embeddings
		embResolved, err := resolveOptions(
			variableNamePrefix+"EMBEDDINGS_",
			provider.NewEmbeddingsProviderOptions,
		)
		if err != nil {
			return errors.Wrap(err, "could not resolve embeddings options")
		}
		opts.Embeddings = embResolved

		return nil
	}
}

// resolveOptions effectue les deux passes de parsing pour un type de client donné.
func resolveOptions(
	typePrefix string,
	newProviderOptions func(provider.Name) any,
) (*provider.ResolvedClientOptions, error) {
	// Passe 1 : identifier le provider actif
	clientOpts := &provider.ClientOptions{}
	if err := env.ParseWithOptions(clientOpts, env.Options{Prefix: typePrefix}); err != nil {
		return nil, errors.Wrap(err, "could not parse provider selection")
	}

	if clientOpts.Provider == "" {
		return nil, nil
	}

	// Passe 2 : instancier et peupler les options spécifiques au provider
	// Si le provider n'est pas enregistré, specificOpts == nil →
	// les variables d'env spécifiques sont ignorées silencieusement.
	specificOpts := newProviderOptions(clientOpts.Provider)
	if specificOpts != nil {
		// Les tirets dans le nom du provider sont remplacés par des underscores pour produire
		// un nom de variable d'environnement POSIX valide (ex. "my-provider" → "MY_PROVIDER").
		providerPrefix := typePrefix + strings.ReplaceAll(strings.ToUpper(string(clientOpts.Provider)), "-", "_") + "_"
		if err := env.ParseWithOptions(specificOpts, env.Options{Prefix: providerPrefix}); err != nil {
			return nil, errors.Wrapf(err, "could not parse options for provider '%s'", clientOpts.Provider)
		}
	}

	return &provider.ResolvedClientOptions{
		Provider: clientOpts.Provider,
		Specific: specificOpts,
	}, nil
}

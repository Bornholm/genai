package provider

import "github.com/pkg/errors"

// ClientOptions sert uniquement à identifier le provider actif (passe 1 de env.With).
type ClientOptions struct {
	Provider Name `env:"PROVIDER"`
}

// CommonOptions contient les options HTTP communes embeddables dans les structs provider.
// Avec caarlos0/env, les champs d'une struct embeddée héritent du préfixe courant.
type CommonOptions struct {
	Model   string `env:"MODEL"`
	BaseURL string `env:"BASE_URL"`
	APIKey  string `env:"API_KEY"`
}

// ResolvedClientOptions transporte le provider identifié et ses options spécifiques
// entre env.With et Registry.Create.
type ResolvedClientOptions struct {
	Provider Name
	Specific any // *T : pointeur vers struct d'options du provider
}

// Options regroupe les options résolues pour chat completion et embeddings.
type Options struct {
	ChatCompletion *ResolvedClientOptions
	Embeddings     *ResolvedClientOptions
}

// Validator est une interface optionnelle que les structs d'options peuvent implémenter.
// Registry.Create appelle Validate() avant de passer les options à la factory.
type Validator interface {
	Validate() error
}

// OptionFunc est une fonction qui configure Options.
type OptionFunc func(opts *Options) error

// NewOptions applique les OptionFunc et retourne les options résultantes.
func NewOptions(funcs ...OptionFunc) (*Options, error) {
	opts := &Options{}
	for _, fn := range funcs {
		if err := fn(opts); err != nil {
			return nil, errors.WithStack(err)
		}
	}
	return opts, nil
}

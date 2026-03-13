package provider

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

var (
	ErrClientNotFound = errors.New("not found")
	ErrNotConfigured  = errors.New("not configured")
)

var defaultRegistry = newRegistry()

// Name identifie un provider.
type Name string

// providerEntry stocke les fonctions d'options et de création d'un provider.
// newOptions retourne TOUJOURS un *T (pointeur vers struct), jamais une valeur.
// createClient effectue le type assertion opts.(*T) — une panique indique une
// erreur d'implémentation du provider (newOptions() retournant nil ou une valeur non-pointeur).
type providerEntry struct {
	newOptions   func() any
	createClient func(ctx context.Context, opts any) (any, error)
}

type Registry struct {
	chatCompletionEntries map[Name]providerEntry
	embeddingsEntries     map[Name]providerEntry
}

// RegisterChatCompletion enregistre un provider de chat completion dans le registry global.
// newOptions doit retourner un *T non-nil avec les valeurs par défaut.
// factory reçoit le *T peuplé depuis les variables d'environnement.
func RegisterChatCompletion[T any](
	name Name,
	newOptions func() *T,
	factory func(ctx context.Context, opts *T) (llm.ChatCompletionClient, error),
) {
	defaultRegistry.chatCompletionEntries[name] = providerEntry{
		newOptions: func() any { return newOptions() },
		createClient: func(ctx context.Context, opts any) (any, error) {
			return factory(ctx, opts.(*T))
		},
	}
}

// RegisterEmbeddings enregistre un provider d'embeddings dans le registry global.
func RegisterEmbeddings[T any](
	name Name,
	newOptions func() *T,
	factory func(ctx context.Context, opts *T) (llm.EmbeddingsClient, error),
) {
	defaultRegistry.embeddingsEntries[name] = providerEntry{
		newOptions: func() any { return newOptions() },
		createClient: func(ctx context.Context, opts any) (any, error) {
			return factory(ctx, opts.(*T))
		},
	}
}

// NewChatCompletionProviderOptions retourne une instance d'options (avec les defaults)
// pour le provider de chat completion donné, ou nil si le provider n'est pas enregistré.
func NewChatCompletionProviderOptions(name Name) any {
	if entry, ok := defaultRegistry.chatCompletionEntries[name]; ok {
		return entry.newOptions()
	}
	return nil
}

// NewEmbeddingsProviderOptions retourne une instance d'options (avec les defaults)
// pour le provider d'embeddings donné, ou nil si le provider n'est pas enregistré.
func NewEmbeddingsProviderOptions(name Name) any {
	if entry, ok := defaultRegistry.embeddingsEntries[name]; ok {
		return entry.newOptions()
	}
	return nil
}

// Create crée un llm.Client à partir des options résolues.
func (r *Registry) Create(ctx context.Context, funcs ...OptionFunc) (llm.Client, error) {
	opts, err := NewOptions(funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	chatCompletion, err := createClientFromResolved[llm.ChatCompletionClient](ctx, opts.ChatCompletion, r.chatCompletionEntries)
	if err != nil && !errors.Is(err, ErrNotConfigured) {
		return nil, errors.WithStack(err)
	}

	embeddings, err := createClientFromResolved[llm.EmbeddingsClient](ctx, opts.Embeddings, r.embeddingsEntries)
	if err != nil && !errors.Is(err, ErrNotConfigured) {
		return nil, errors.WithStack(err)
	}

	if chatCompletion == nil && embeddings == nil {
		return nil, errors.WithStack(ErrNotConfigured)
	}

	return NewClient(chatCompletion, embeddings), nil
}

// createClientFromResolved crée un client T à partir des options résolues.
func createClientFromResolved[T any](
	ctx context.Context,
	resolved *ResolvedClientOptions,
	entries map[Name]providerEntry,
) (T, error) {
	var zero T

	if resolved == nil {
		return zero, errors.WithStack(ErrNotConfigured)
	}

	if resolved.Provider == "" {
		return zero, llm.NewValidationError("provider", "provider is required")
	}

	entry, exists := entries[resolved.Provider]
	if !exists {
		return zero, errors.Wrapf(ErrClientNotFound, "could not find client factory for provider '%s'", resolved.Provider)
	}

	if resolved.Specific != nil {
		if v, ok := resolved.Specific.(Validator); ok {
			if err := v.Validate(); err != nil {
				return zero, errors.Wrapf(err, "invalid options for provider '%s'", resolved.Provider)
			}
		}
	}

	result, err := entry.createClient(ctx, resolved.Specific)
	if err != nil {
		return zero, errors.Wrapf(err, "could not create client with provider '%s'", resolved.Provider)
	}

	return result.(T), nil
}

func newRegistry() *Registry {
	return &Registry{
		chatCompletionEntries: map[Name]providerEntry{},
		embeddingsEntries:     map[Name]providerEntry{},
	}
}

// Create est la fonction globale qui délègue au defaultRegistry.
func Create(ctx context.Context, funcs ...OptionFunc) (llm.Client, error) {
	return defaultRegistry.Create(ctx, funcs...)
}

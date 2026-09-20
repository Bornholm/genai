package provider

import (
	"context"
	"io"
	"sync"

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

// Capability names one of the client kinds a provider can offer.
type Capability string

const (
	CapabilityChatCompletion  Capability = "chat_completion"
	CapabilityEmbeddings      Capability = "embeddings"
	CapabilityTranscription   Capability = "transcription"
	CapabilityImageGeneration Capability = "image_generation"
)

// FallbackEntry is what a FallbackFunc returns for a provider it can serve.
// NewOptions must return a non-nil pointer to a struct; CreateClient receives
// that pointer back, populated, and returns a client implementing the
// interface of the requested capability.
type FallbackEntry struct {
	NewOptions   func() any
	CreateClient func(ctx context.Context, opts any) (any, error)
}

// FallbackFunc resolves a provider name that no in-process registration
// covers. It is how dynamically loaded providers plug into the registry: the
// registry asks each fallback, in registration order, and uses the first one
// that answers.
type FallbackFunc func(capability Capability, name Name) (*FallbackEntry, bool)

// RawEnvConsumer is implemented by option types that want every environment
// variable under their provider prefix, not only the ones matching a struct
// field. A dynamically loaded provider validates its own options, so the host
// forwards them verbatim.
type RawEnvConsumer interface {
	SetRawEnv(vars map[string]string)
}

type Registry struct {
	chatCompletionEntries  map[Name]providerEntry
	embeddingsEntries      map[Name]providerEntry
	transcriptionEntries   map[Name]providerEntry
	imageGenerationEntries map[Name]providerEntry

	fallbacksMu sync.RWMutex
	fallbacks   []FallbackFunc
}

// RegisterFallback adds a resolver consulted for provider names that have no
// in-process registration, in the global registry. Unlike the Register*
// functions, it may be called while the registry is in use.
func RegisterFallback(fn FallbackFunc) {
	defaultRegistry.fallbacksMu.Lock()
	defer defaultRegistry.fallbacksMu.Unlock()
	defaultRegistry.fallbacks = append(defaultRegistry.fallbacks, fn)
}

func (r *Registry) entries(capability Capability) map[Name]providerEntry {
	switch capability {
	case CapabilityChatCompletion:
		return r.chatCompletionEntries
	case CapabilityEmbeddings:
		return r.embeddingsEntries
	case CapabilityTranscription:
		return r.transcriptionEntries
	case CapabilityImageGeneration:
		return r.imageGenerationEntries
	default:
		return nil
	}
}

// lookup finds the entry for a provider, in-process registrations first, then
// fallbacks in registration order.
func (r *Registry) lookup(capability Capability, name Name) (providerEntry, bool) {
	if entry, ok := r.entries(capability)[name]; ok {
		return entry, true
	}
	r.fallbacksMu.RLock()
	fallbacks := r.fallbacks
	r.fallbacksMu.RUnlock()
	for _, fn := range fallbacks {
		fallback, ok := fn(capability, name)
		if !ok || fallback == nil {
			continue
		}
		return providerEntry{
			newOptions:   fallback.NewOptions,
			createClient: fallback.CreateClient,
		}, true
	}
	return providerEntry{}, false
}

// newOptions returns a fresh options value for the provider, or nil when it is
// unknown to both the in-process registrations and the fallbacks.
func (r *Registry) newOptions(capability Capability, name Name) any {
	if entry, ok := r.lookup(capability, name); ok {
		return entry.newOptions()
	}
	return nil
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

// RegisterTranscription enregistre un provider de transcription audio dans le registry global.
func RegisterTranscription[T any](
	name Name,
	newOptions func() *T,
	factory func(ctx context.Context, opts *T) (llm.TranscriptionClient, error),
) {
	defaultRegistry.transcriptionEntries[name] = providerEntry{
		newOptions: func() any { return newOptions() },
		createClient: func(ctx context.Context, opts any) (any, error) {
			return factory(ctx, opts.(*T))
		},
	}
}

// RegisterImageGeneration enregistre un provider de génération d'images dans le registry global.
func RegisterImageGeneration[T any](
	name Name,
	newOptions func() *T,
	factory func(ctx context.Context, opts *T) (llm.ImageGenerationClient, error),
) {
	defaultRegistry.imageGenerationEntries[name] = providerEntry{
		newOptions: func() any { return newOptions() },
		createClient: func(ctx context.Context, opts any) (any, error) {
			return factory(ctx, opts.(*T))
		},
	}
}

// NewImageGenerationProviderOptions retourne une instance d'options (avec les defaults)
// pour le provider de génération d'images donné, ou nil si le provider n'est pas enregistré.
func NewImageGenerationProviderOptions(name Name) any {
	return defaultRegistry.newOptions(CapabilityImageGeneration, name)
}

// NewChatCompletionProviderOptions retourne une instance d'options (avec les defaults)
// pour le provider de chat completion donné, ou nil si le provider n'est pas enregistré.
func NewChatCompletionProviderOptions(name Name) any {
	return defaultRegistry.newOptions(CapabilityChatCompletion, name)
}

// NewEmbeddingsProviderOptions retourne une instance d'options (avec les defaults)
// pour le provider d'embeddings donné, ou nil si le provider n'est pas enregistré.
func NewEmbeddingsProviderOptions(name Name) any {
	return defaultRegistry.newOptions(CapabilityEmbeddings, name)
}

// NewTranscriptionProviderOptions retourne une instance d'options (avec les defaults)
// pour le provider de transcription donné, ou nil si le provider n'est pas enregistré.
func NewTranscriptionProviderOptions(name Name) any {
	return defaultRegistry.newOptions(CapabilityTranscription, name)
}

// Create crée un llm.Client à partir des options résolues.
func (r *Registry) Create(ctx context.Context, funcs ...OptionFunc) (llm.Client, error) {
	opts, err := NewOptions(funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	// A client created before a later capability fails would leak what it
	// holds (a plugin client keeps a configured instance, possibly a loaded
	// model, in its process): close the ones already built on the way out.
	var created []any
	defer func() {
		if err == nil {
			return
		}
		for _, client := range created {
			if closer, ok := client.(io.Closer); ok {
				_ = closer.Close()
			}
		}
	}()

	chatCompletion, err := createClientFromResolved[llm.ChatCompletionClient](ctx, r, CapabilityChatCompletion, opts.ChatCompletion)
	if err != nil && !errors.Is(err, ErrNotConfigured) {
		return nil, errors.WithStack(err)
	}
	created = append(created, chatCompletion)

	embeddings, err := createClientFromResolved[llm.EmbeddingsClient](ctx, r, CapabilityEmbeddings, opts.Embeddings)
	if err != nil && !errors.Is(err, ErrNotConfigured) {
		return nil, errors.WithStack(err)
	}
	created = append(created, embeddings)

	transcription, err := createClientFromResolved[llm.TranscriptionClient](ctx, r, CapabilityTranscription, opts.Transcription)
	if err != nil && !errors.Is(err, ErrNotConfigured) {
		return nil, errors.WithStack(err)
	}
	created = append(created, transcription)

	imageGeneration, err := createClientFromResolved[llm.ImageGenerationClient](ctx, r, CapabilityImageGeneration, opts.ImageGeneration)
	if err != nil && !errors.Is(err, ErrNotConfigured) {
		return nil, errors.WithStack(err)
	}

	if chatCompletion == nil && embeddings == nil && transcription == nil && imageGeneration == nil {
		err = errors.WithStack(ErrNotConfigured)
		return nil, err
	}

	err = nil
	return NewClientWithImageGeneration(chatCompletion, embeddings, transcription, imageGeneration), nil
}

// createClientFromResolved crée un client T à partir des options résolues.
func createClientFromResolved[T any](
	ctx context.Context,
	r *Registry,
	capability Capability,
	resolved *ResolvedClientOptions,
) (T, error) {
	var zero T

	if resolved == nil {
		return zero, errors.WithStack(ErrNotConfigured)
	}

	if resolved.Provider == "" {
		return zero, llm.NewValidationError("provider", "provider is required")
	}

	entry, exists := r.lookup(capability, resolved.Provider)
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
		chatCompletionEntries:  map[Name]providerEntry{},
		embeddingsEntries:      map[Name]providerEntry{},
		transcriptionEntries:   map[Name]providerEntry{},
		imageGenerationEntries: map[Name]providerEntry{},
	}
}

// Create est la fonction globale qui délègue au defaultRegistry.
func Create(ctx context.Context, funcs ...OptionFunc) (llm.Client, error) {
	return defaultRegistry.Create(ctx, funcs...)
}

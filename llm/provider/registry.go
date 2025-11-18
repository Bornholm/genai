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

var defaultRegistry = NewRegistry()

type Name string

type Registry struct {
	chatCompletionFactories map[Name]Factory[llm.ChatCompletionClient]
	embeddingsFactories     map[Name]Factory[llm.EmbeddingsClient]
}

type Factory[T any] func(ctx context.Context, opts ClientOptions) (T, error)

type ContextFunc func(ctx context.Context) context.Context

func (r *Registry) RegisterChatCompletion(name Name, factory Factory[llm.ChatCompletionClient]) {
	r.chatCompletionFactories[name] = factory
}

func (r *Registry) RegisterEmbeddings(name Name, factory Factory[llm.EmbeddingsClient]) {
	r.embeddingsFactories[name] = factory
}

func (r *Registry) Create(ctx context.Context, funcs ...OptionFunc) (llm.Client, error) {
	opts, err := NewOptions(funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	chatCompletion, err := createClient(ctx, opts.ChatCompletion, r.chatCompletionFactories)
	if err != nil && !errors.Is(err, ErrNotConfigured) {
		return nil, errors.WithStack(err)
	}

	embeddings, err := createClient(ctx, opts.Embeddings, r.embeddingsFactories)
	if err != nil && !errors.Is(err, ErrNotConfigured) {
		return nil, errors.WithStack(err)
	}

	if embeddings == nil && chatCompletion == nil {
		return nil, errors.WithStack(ErrNotConfigured)
	}

	return NewClient(chatCompletion, embeddings), nil
}

func createClient[T any](ctx context.Context, clientOpts *ClientOptions, factories map[Name]Factory[T]) (T, error) {
	var zero T

	if clientOpts == nil {
		return zero, errors.WithStack(ErrNotConfigured)
	}

	// Validate client options
	if err := clientOpts.Validate(); err != nil {
		return zero, errors.WithStack(err)
	}

	provider := clientOpts.Provider

	clientFactory, exists := factories[provider]
	if !exists {
		return zero, errors.Wrapf(ErrClientNotFound, "could not find client factory for provider '%s'", clientOpts.Provider)
	}

	client, err := clientFactory(ctx, *clientOpts)
	if err != nil {
		return zero, errors.Wrapf(err, "could not create client with provider '%s'", clientOpts.Provider)
	}

	return client, nil
}

func NewRegistry() *Registry {
	return &Registry{
		chatCompletionFactories: map[Name]Factory[llm.ChatCompletionClient]{},
		embeddingsFactories:     map[Name]Factory[llm.EmbeddingsClient]{},
	}
}

func RegisterChatCompletion(name Name, factory Factory[llm.ChatCompletionClient]) {
	defaultRegistry.RegisterChatCompletion(name, factory)
}

func RegisterEmbeddings(name Name, factory Factory[llm.EmbeddingsClient]) {
	defaultRegistry.RegisterEmbeddings(name, factory)
}

func Create(ctx context.Context, funcs ...OptionFunc) (llm.Client, error) {
	return defaultRegistry.Create(ctx, funcs...)
}

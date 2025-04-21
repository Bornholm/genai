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
	extractTextFactories    map[Name]Factory[llm.ExtractTextClient]
}

type Factory[T any] func(ctx context.Context, opts ClientOptions) (T, error)

type ContextFunc func(ctx context.Context) context.Context

func (r *Registry) RegisterChatCompletion(name Name, factory Factory[llm.ChatCompletionClient]) {
	r.chatCompletionFactories[name] = factory
}

func (r *Registry) RegisterEmbeddings(name Name, factory Factory[llm.EmbeddingsClient]) {
	r.embeddingsFactories[name] = factory
}

func (r *Registry) RegisterExtractText(name Name, factory Factory[llm.ExtractTextClient]) {
	r.extractTextFactories[name] = factory
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

	extractText, err := createClient(ctx, opts.ExtractText, r.extractTextFactories)
	if err != nil && !errors.Is(err, ErrNotConfigured) {
		return nil, errors.WithStack(err)
	}

	return NewClient(chatCompletion, embeddings, extractText), nil
}

func createClient[T any](ctx context.Context, clientOpts *ClientOptions, factories map[Name]Factory[T]) (T, error) {
	if clientOpts == nil {
		return *new(T), errors.WithStack(ErrNotConfigured)
	}

	provider := clientOpts.Provider

	clientFactory, exists := factories[provider]
	if !exists {
		return *new(T), errors.Wrapf(ErrClientNotFound, "could not find client factory for provider '%s'", clientOpts.Provider)
	}

	client, err := clientFactory(ctx, *clientOpts)
	if err != nil {
		return *new(T), errors.Wrapf(err, "could not create client with provider '%s'", clientOpts.Provider)
	}

	return client, nil
}

func NewRegistry() *Registry {
	return &Registry{
		chatCompletionFactories: map[Name]Factory[llm.ChatCompletionClient]{},
		embeddingsFactories:     map[Name]Factory[llm.EmbeddingsClient]{},
		extractTextFactories:    map[Name]Factory[llm.ExtractTextClient]{},
	}
}

func RegisterChatCompletion(name Name, factory Factory[llm.ChatCompletionClient]) {
	defaultRegistry.RegisterChatCompletion(name, factory)
}

func RegisterEmbeddings(name Name, factory Factory[llm.EmbeddingsClient]) {
	defaultRegistry.RegisterEmbeddings(name, factory)
}

func RegisterExtractText(name Name, factory Factory[llm.ExtractTextClient]) {
	defaultRegistry.RegisterExtractText(name, factory)
}

func Create(ctx context.Context, funcs ...OptionFunc) (llm.Client, error) {
	return defaultRegistry.Create(ctx, funcs...)
}

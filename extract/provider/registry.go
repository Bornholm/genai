package provider

import (
	"context"
	"net/url"

	"github.com/bornholm/genai/extract"
	"github.com/pkg/errors"
)

var (
	ErrClientNotFound = errors.New("not found")
	ErrNotConfigured  = errors.New("not configured")
)

var defaultRegistry = NewRegistry()

type Name string

type Registry struct {
	textFactories map[Name]Factory[extract.TextClient]
}

type Factory[T any] func(ctx context.Context, dsn *url.URL) (T, error)

type ContextFunc func(ctx context.Context) context.Context

func (r *Registry) RegisterTextClient(name Name, factory Factory[extract.TextClient]) {
	r.textFactories[name] = factory
}

func (r *Registry) Create(ctx context.Context, funcs ...OptionFunc) (extract.Client, error) {
	opts, err := NewOptions(funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	textClient, err := createClient(ctx, opts.TextDSN, r.textFactories)
	if err != nil && !errors.Is(err, ErrNotConfigured) {
		return nil, errors.WithStack(err)
	}

	return NewClient(textClient), nil
}

func createClient[T any](ctx context.Context, rawDSN string, factories map[Name]Factory[T]) (T, error) {
	if rawDSN == "" {
		return *new(T), errors.WithStack(ErrNotConfigured)
	}

	dsn, err := url.Parse(rawDSN)
	if err != nil {
		return *new(T), errors.WithStack(err)
	}

	provider := Name(dsn.Scheme)

	clientFactory, exists := factories[provider]
	if !exists {
		return *new(T), errors.Wrapf(ErrClientNotFound, "could not find client factory for provider '%s'", provider)
	}

	client, err := clientFactory(ctx, dsn)
	if err != nil {
		return *new(T), errors.Wrapf(err, "could not create client with provider '%s'", provider)
	}

	return client, nil
}

func NewRegistry() *Registry {
	return &Registry{
		textFactories: map[Name]Factory[extract.TextClient]{},
	}
}

func RegisterTextClient(name Name, factory Factory[extract.TextClient]) {
	defaultRegistry.RegisterTextClient(name, factory)
}

func Create(ctx context.Context, funcs ...OptionFunc) (extract.Client, error) {
	return defaultRegistry.Create(ctx, funcs...)
}

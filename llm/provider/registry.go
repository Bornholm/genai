package provider

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

var (
	ErrClientNotFound = errors.New("not found")
)

var defaultRegistry = NewRegistry()

type Name string

type Registry struct {
	factories map[Name]Factory
}

type Factory func(ctx context.Context) (llm.Client, error)

func (r *Registry) Register(name Name, factory Factory) {
	r.factories[name] = factory
}

func (r *Registry) Create(ctx context.Context, name Name) (llm.Client, error) {
	factory, exists := r.factories[name]
	if !exists {
		return nil, errors.WithStack(ErrClientNotFound)
	}

	client, err := factory(ctx)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return client, nil
}

func NewRegistry() *Registry {
	return &Registry{
		factories: make(map[Name]Factory),
	}
}

func Register(name Name, factory Factory) {
	defaultRegistry.Register(name, factory)
}

func Create(ctx context.Context, name Name) (llm.Client, error) {
	return defaultRegistry.Create(ctx, name)
}

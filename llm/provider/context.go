package provider

import (
	"context"
	"os"

	"github.com/pkg/errors"
)

type ContextKey string

var ErrContextKeyNotFound = errors.New("not found")

const (
	ContextKeyAPIProvider ContextKey = "LLM_API_PROVIDER"
	ContextKeyAPIBaseURL  ContextKey = "LLM_API_BASE_URL"
	ContextKeyAPIKey      ContextKey = "LLM_API_KEY"
	ContextKeyModel       ContextKey = "LLM_MODEL"
)

func FromMap(ctx context.Context, prefix string, values map[string]string) context.Context {
	if provider, exists := values[prefix+string(ContextKeyAPIProvider)]; exists {
		ctx = WithAPIProvider(ctx, Name(provider))
	}

	if baseURL, exists := values[prefix+string(ContextKeyAPIBaseURL)]; exists {
		ctx = WithAPIBaseURL(ctx, baseURL)
	}

	if apiKey, exists := values[prefix+string(ContextKeyAPIKey)]; exists {
		ctx = WithAPIKey(ctx, apiKey)
	}

	if model, exists := values[prefix+string(ContextKeyModel)]; exists {
		ctx = WithModel(ctx, model)
	}

	return ctx
}

func FromEnvironment(ctx context.Context, prefix string) context.Context {
	if provider, exists := os.LookupEnv(prefix + string(ContextKeyAPIProvider)); exists {
		ctx = WithAPIProvider(ctx, Name(provider))
	}

	if baseURL, exists := os.LookupEnv(prefix + string(ContextKeyAPIBaseURL)); exists {
		ctx = WithAPIBaseURL(ctx, baseURL)
	}

	if apiKey, exists := os.LookupEnv(prefix + string(ContextKeyAPIKey)); exists {
		ctx = WithAPIKey(ctx, apiKey)
	}

	if model, exists := os.LookupEnv(prefix + string(ContextKeyModel)); exists {
		ctx = WithModel(ctx, model)
	}

	return ctx
}

func WithAPIProvider(ctx context.Context, provider Name) context.Context {
	return context.WithValue(ctx, ContextKeyAPIProvider, provider)
}

func ContextAPIProvider(ctx context.Context) (Name, error) {
	return contextValue[Name](ctx, ContextKeyAPIProvider)
}

func WithAPIBaseURL(ctx context.Context, baseURL string) context.Context {
	return context.WithValue(ctx, ContextKeyAPIBaseURL, baseURL)
}

func ContextAPIBaseURL(ctx context.Context) (string, error) {
	return contextValue[string](ctx, ContextKeyAPIBaseURL)
}

func WithAPIKey(ctx context.Context, apiKey string) context.Context {
	return context.WithValue(ctx, ContextKeyAPIKey, apiKey)
}

func ContextAPIKey(ctx context.Context) (string, error) {
	return contextValue[string](ctx, ContextKeyAPIKey)
}

func WithModel(ctx context.Context, model string) context.Context {
	return context.WithValue(ctx, ContextKeyModel, model)
}

func ContextModel(ctx context.Context) (string, error) {
	return contextValue[string](ctx, ContextKeyModel)
}

func contextValue[T any](ctx context.Context, key ContextKey) (T, error) {
	raw := ctx.Value(key)
	if raw == nil {
		return *new(T), errors.WithStack(ErrContextKeyNotFound)
	}

	value, ok := raw.(T)
	if !ok {
		return *new(T), errors.Errorf("unexpected type '%T' for context value '%s'", raw, key)
	}

	return value, nil
}

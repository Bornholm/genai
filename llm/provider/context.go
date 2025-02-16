package provider

import (
	"context"
	"os"

	"github.com/pkg/errors"
)

type contextKey string

var ErrContextKeyNotFound = errors.New("not found")

const (
	contextKeyAPIBaseURL contextKey = "LLM_API_BASE_URL"
	contextKeyAPIKey     contextKey = "LLM_API_KEY"
	contextModel         contextKey = "LLM_MODEL"
)

func FromMap(ctx context.Context, prefix string, values map[string]string) context.Context {
	if baseURL, exists := values[prefix+string(contextKeyAPIBaseURL)]; exists {
		ctx = WithAPIBaseURL(ctx, baseURL)
	}

	if apiKey, exists := values[prefix+string(contextKeyAPIKey)]; exists {
		ctx = WithAPIKey(ctx, apiKey)
	}

	if model, exists := values[prefix+string(contextModel)]; exists {
		ctx = WithModel(ctx, model)
	}

	return ctx
}

func FromEnvironment(ctx context.Context, prefix string) context.Context {
	if baseURL, exists := os.LookupEnv(prefix + string(contextKeyAPIBaseURL)); exists {
		ctx = WithAPIBaseURL(ctx, baseURL)
	}

	if apiKey, exists := os.LookupEnv(prefix + string(contextKeyAPIKey)); exists {
		ctx = WithAPIKey(ctx, apiKey)
	}

	if model, exists := os.LookupEnv(prefix + string(contextModel)); exists {
		ctx = WithModel(ctx, model)
	}

	return ctx
}

func WithAPIBaseURL(ctx context.Context, baseURL string) context.Context {
	return context.WithValue(ctx, contextKeyAPIBaseURL, baseURL)
}

func ContextAPIBaseURL(ctx context.Context) (string, error) {
	return contextValue[string](ctx, contextKeyAPIBaseURL)
}

func WithAPIKey(ctx context.Context, apiKey string) context.Context {
	return context.WithValue(ctx, contextKeyAPIKey, apiKey)
}

func ContextAPIKey(ctx context.Context) (string, error) {
	return contextValue[string](ctx, contextKeyAPIKey)
}

func WithModel(ctx context.Context, model string) context.Context {
	return context.WithValue(ctx, contextModel, model)
}

func ContextModel(ctx context.Context) (string, error) {
	return contextValue[string](ctx, contextModel)
}

func contextValue[T any](ctx context.Context, key contextKey) (T, error) {
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

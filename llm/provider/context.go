package provider

import (
	"context"
	"os"

	"github.com/pkg/errors"
)

type contextKey string

var ErrContextKeyNotFound = errors.New("not found")

const (
	ContextKeyProvider            contextKey = "PROVIDER"
	ContextKeyBaseURL             contextKey = "BASE_URL"
	ContextKeyKey                 contextKey = "KEY"
	ContextKeyChatCompletionModel contextKey = "CHAT_COMPLETION_MODEL"
	ContextKeyEmbeddingsModel     contextKey = "EMBEDDINGS_MODEL"
)

type Config struct {
	Provider            Name
	BaseURL             string
	Key                 string
	ChatCompletionModel string
	EmbeddingsModel     string
}

func WithConfig(conf *Config) ContextFunc {
	return func(ctx context.Context) context.Context {
		ctx = WithProvider(conf.Provider)(ctx)
		ctx = WithBaseURL(conf.BaseURL)(ctx)
		ctx = WithKey(conf.Key)(ctx)
		ctx = WithChatCompletionModel(conf.ChatCompletionModel)(ctx)
		ctx = WithEmbeddingsModel(conf.EmbeddingsModel)(ctx)
		return ctx
	}
}

func WithMap(values map[string]string, prefix string) ContextFunc {
	return func(ctx context.Context) context.Context {
		if provider, exists := values[prefix+string(ContextKeyProvider)]; exists {
			ctx = WithProvider(Name(provider))(ctx)
		}

		if baseURL, exists := values[prefix+string(ContextKeyBaseURL)]; exists {
			ctx = WithBaseURL(baseURL)(ctx)
		}

		if key, exists := values[prefix+string(ContextKeyKey)]; exists {
			ctx = WithKey(key)(ctx)
		}

		if chatCompletionModel, exists := values[prefix+string(ContextKeyChatCompletionModel)]; exists {
			ctx = WithChatCompletionModel(chatCompletionModel)(ctx)
		}

		return ctx
	}
}

func WithEnvironment(prefix string) ContextFunc {
	return func(ctx context.Context) context.Context {
		if provider, exists := os.LookupEnv(prefix + string(ContextKeyProvider)); exists {
			ctx = WithProvider(Name(provider))(ctx)
		}

		if baseURL, exists := os.LookupEnv(prefix + string(ContextKeyBaseURL)); exists {
			ctx = WithBaseURL(baseURL)(ctx)
		}

		if key, exists := os.LookupEnv(prefix + string(ContextKeyKey)); exists {
			ctx = WithKey(key)(ctx)
		}

		if chatCompletionModel, exists := os.LookupEnv(prefix + string(ContextKeyChatCompletionModel)); exists {
			ctx = WithChatCompletionModel(chatCompletionModel)(ctx)
		}

		return ctx
	}
}

func WithProvider(provider Name) ContextFunc {
	return func(ctx context.Context) context.Context {
		return context.WithValue(ctx, ContextKeyProvider, provider)
	}
}

func ContextProvider(ctx context.Context) (Name, error) {
	return ContextValue[Name](ctx, ContextKeyProvider)
}

func WithBaseURL(baseURL string) ContextFunc {
	return func(ctx context.Context) context.Context {
		return context.WithValue(ctx, ContextKeyBaseURL, baseURL)
	}
}

func ContextBaseURL(ctx context.Context) (string, error) {
	return ContextValue[string](ctx, ContextKeyBaseURL)
}

func WithKey(key string) ContextFunc {
	return func(ctx context.Context) context.Context {
		return context.WithValue(ctx, ContextKeyKey, key)
	}
}

func ContextKey(ctx context.Context) (string, error) {
	return ContextValue[string](ctx, ContextKeyKey)
}

func WithChatCompletionModel(model string) ContextFunc {
	return func(ctx context.Context) context.Context {
		return context.WithValue(ctx, ContextKeyChatCompletionModel, model)
	}
}

func ContextChatCompletionModel(ctx context.Context) (string, error) {
	return ContextValue[string](ctx, ContextKeyChatCompletionModel)
}

func WithEmbeddingsModel(model string) ContextFunc {
	return func(ctx context.Context) context.Context {
		return context.WithValue(ctx, ContextKeyEmbeddingsModel, model)
	}
}

func ContextEmbeddingsModel(ctx context.Context) (string, error) {
	return ContextValue[string](ctx, ContextKeyEmbeddingsModel)
}

func ContextValue[T any, C any](ctx context.Context, key C) (T, error) {
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

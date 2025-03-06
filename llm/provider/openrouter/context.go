package openrouter

import (
	"context"

	"github.com/bornholm/genai/llm/provider"
)

type contextKey string

const (
	ContextKeyTransforms contextKey = "transforms"
)

func ContextTransforms(ctx context.Context) ([]string, error) {
	return provider.ContextValue[[]string](ctx, ContextKeyTransforms)
}

func WithTransforms(ctx context.Context, transforms []string) context.Context {
	return context.WithValue(ctx, ContextKeyTransforms, transforms)
}

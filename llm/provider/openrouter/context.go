package openrouter

import "github.com/bornholm/genai/llm/context"

type contextKey string

const (
	ContextKeyTransforms contextKey = "transforms"
)

func ContextTransforms(ctx context.Context) ([]string, error) {
	return context.Value[[]string](ctx, ContextKeyTransforms)
}

func WithTransforms(ctx context.Context, transforms []string) context.Context {
	return context.WithValue(ctx, ContextKeyTransforms, transforms)
}

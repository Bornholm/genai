package openrouter

import "github.com/bornholm/genai/llm/context"

type contextKey string

const (
	contextKeyTransforms contextKey = "transforms"
	contextKeyModels     contextKey = "models"
)

func ContextTransforms(ctx context.Context) ([]string, error) {
	return context.Value[[]string](ctx, contextKeyTransforms)
}

func WithTransforms(ctx context.Context, transforms []string) context.Context {
	return context.WithValue(ctx, contextKeyTransforms, transforms)
}

func ContextModels(ctx context.Context) ([]string, error) {
	return context.Value[[]string](ctx, contextKeyModels)
}

func WithModels(ctx context.Context, models ...string) context.Context {
	return context.WithValue(ctx, contextKeyModels, models)
}

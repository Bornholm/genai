package task

import (
	"context"

	"github.com/bornholm/genai/llm"
)

type contextKey string

const (
	contextKeyClient        contextKey = "client"
	contextKeyTools         contextKey = "tools"
	contextKeyMaxIterations contextKey = "maxIterations"
	contextKeyEvaluator     contextKey = "evaluator"
)

func WithContextClient(ctx context.Context, client llm.ChatCompletionClient) context.Context {
	return context.WithValue(ctx, contextKeyClient, client)
}

func ContextClient(ctx context.Context, defaultClient llm.ChatCompletionClient) llm.ChatCompletionClient {
	return contextValue(ctx, contextKeyClient, defaultClient)
}

func WithContextTools(ctx context.Context, tools []llm.Tool) context.Context {
	return context.WithValue(ctx, contextKeyTools, tools)
}

func ContextTools(ctx context.Context, defaultTools []llm.Tool) []llm.Tool {
	return contextValue(ctx, contextKeyTools, defaultTools)
}

func WithContextMaxIterations(ctx context.Context, maxIterations int) context.Context {
	return context.WithValue(ctx, contextKeyMaxIterations, maxIterations)
}

func ContextMaxIterations(ctx context.Context, defaultMaxIterations int) int {
	return contextValue(ctx, contextKeyMaxIterations, defaultMaxIterations)
}

func WithContextEvaluator(ctx context.Context, evaluator Evaluator) context.Context {
	return context.WithValue(ctx, contextKeyEvaluator, evaluator)
}

func ContextEvaluator(ctx context.Context, defaultEvaluator Evaluator) Evaluator {
	return contextValue(ctx, contextKeyEvaluator, defaultEvaluator)
}

func contextValue[T any](ctx context.Context, key contextKey, defaultValue T) T {
	raw := ctx.Value(key)
	if raw == nil {
		return defaultValue
	}

	value, ok := raw.(T)
	if !ok {
		return value
	}

	return value
}

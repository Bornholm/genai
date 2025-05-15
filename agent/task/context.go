package task

import (
	"context"

	"github.com/bornholm/genai/agent"
)

type contextKey string

const (
	contextKeyMinIterations contextKey = "minIterations"
	contextKeyMaxIterations contextKey = "maxIterations"
	contextKeyEvaluator     contextKey = "evaluator"
)

func WithContextMinIterations(ctx context.Context, minIterations int) context.Context {
	return context.WithValue(ctx, contextKeyMinIterations, minIterations)
}

func ContextMinIterations(ctx context.Context, defaultMinIterations int) int {
	return agent.ContextValue(ctx, contextKeyMinIterations, defaultMinIterations)
}

func WithContextMaxIterations(ctx context.Context, maxIterations int) context.Context {
	return context.WithValue(ctx, contextKeyMaxIterations, maxIterations)
}

func ContextMaxIterations(ctx context.Context, defaultMaxIterations int) int {
	return agent.ContextValue(ctx, contextKeyMaxIterations, defaultMaxIterations)
}

func WithContextEvaluator(ctx context.Context, evaluator Evaluator) context.Context {
	return context.WithValue(ctx, contextKeyEvaluator, evaluator)
}

func ContextEvaluator(ctx context.Context, defaultEvaluator Evaluator) Evaluator {
	return agent.ContextValue(ctx, contextKeyEvaluator, defaultEvaluator)
}

package task

import (
	"context"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/llm"
)

const (
	contextKeyMinIterations agent.ContextKey = "minIterations"
	contextKeyMaxIterations agent.ContextKey = "maxIterations"
	contextKeyEvaluator     agent.ContextKey = "evaluator"
	contextKeySchema        agent.ContextKey = "schema"
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

func WithContextSchema(ctx context.Context, schema llm.ResponseSchema) context.Context {
	return context.WithValue(ctx, contextKeySchema, schema)
}

func ContextSchema(ctx context.Context, defaultSchema llm.ResponseSchema) llm.ResponseSchema {
	return agent.ContextValue(ctx, contextKeySchema, defaultSchema)
}

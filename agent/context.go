package agent

import (
	"context"

	"github.com/bornholm/genai/llm"
)

type ContextKey string

const (
	ContextKeyAgent       ContextKey = "agent"
	ContextKeyClient      ContextKey = "client"
	ContextKeyTools       ContextKey = "tools"
	ContextKeyMessages    ContextKey = "messages"
	ContextKeyTemperature ContextKey = "temperature"
	ContextKeySeed        ContextKey = "seed"
	ContextKeyError       ContextKey = "error"
)

func WithContextClient(ctx context.Context, client llm.ChatCompletionClient) context.Context {
	return context.WithValue(ctx, ContextKeyClient, client)
}

func ContextClient(ctx context.Context, defaultClient llm.ChatCompletionClient) llm.ChatCompletionClient {
	client := ContextValue(ctx, ContextKeyClient, defaultClient)
	if client == nil {
		return defaultClient
	}

	return client
}

func WithContextTemperature(ctx context.Context, temperature float64) context.Context {
	return context.WithValue(ctx, ContextKeyTemperature, temperature)
}

func ContextTemperature(ctx context.Context, defaultTemperature float64) float64 {
	return ContextValue(ctx, ContextKeyTemperature, defaultTemperature)
}

func WithContextSeed(ctx context.Context, seed int) context.Context {
	return context.WithValue(ctx, ContextKeySeed, seed)
}

func ContextSeed(ctx context.Context, defaultSeed int) int {
	return ContextValue(ctx, ContextKeySeed, defaultSeed)
}

func WithContextTools(ctx context.Context, tools []llm.Tool) context.Context {
	return context.WithValue(ctx, ContextKeyTools, tools)
}

func ContextTools(ctx context.Context, defaultTools []llm.Tool) []llm.Tool {
	return ContextValue(ctx, ContextKeyTools, defaultTools)
}

func WithContextMessages(ctx context.Context, messages []llm.Message) context.Context {
	return context.WithValue(ctx, ContextKeyMessages, messages)
}

func ContextMessages(ctx context.Context, defaultMessages []llm.Message) []llm.Message {
	return ContextValue(ctx, ContextKeyMessages, defaultMessages)
}

func WithContextAgent(ctx context.Context, agent *Agent) context.Context {
	return context.WithValue(ctx, ContextKeyAgent, agent)
}

func ContextAgent(ctx context.Context) *Agent {
	return ContextValue[*Agent](ctx, ContextKeyAgent, nil)
}

func WithContextError(ctx context.Context, err error) context.Context {
	return context.WithValue(ctx, ContextKeyError, err)
}

func ContextError(ctx context.Context) error {
	return ContextValue[error](ctx, ContextKeyError, nil)
}

func ContextValue[T any](ctx context.Context, key ContextKey, defaultValue T) T {
	raw := ctx.Value(key)
	if raw == nil {
		return defaultValue
	}

	value, ok := raw.(T)
	if !ok {
		return defaultValue
	}

	return value
}

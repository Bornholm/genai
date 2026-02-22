package agent

import (
	"context"

	"github.com/bornholm/genai/llm"
)

// ContextKey is the type for context keys in the agent package
type ContextKey string

const (
	ContextKeyClient      ContextKey = "client"
	ContextKeyTools       ContextKey = "tools"
	ContextKeyMessages    ContextKey = "messages"
	ContextKeyTemperature ContextKey = "temperature"
	ContextKeySeed        ContextKey = "seed"
)

// WithContextClient stores a ChatCompletionClient in the context
func WithContextClient(ctx context.Context, client llm.ChatCompletionClient) context.Context {
	return context.WithValue(ctx, ContextKeyClient, client)
}

// ContextClient retrieves a ChatCompletionClient from the context
func ContextClient(ctx context.Context, defaultClient llm.ChatCompletionClient) llm.ChatCompletionClient {
	client := ContextValue(ctx, ContextKeyClient, defaultClient)
	if client == nil {
		return defaultClient
	}

	return client
}

// WithContextTemperature stores a temperature value in the context
func WithContextTemperature(ctx context.Context, temperature float64) context.Context {
	return context.WithValue(ctx, ContextKeyTemperature, temperature)
}

// ContextTemperature retrieves a temperature value from the context
func ContextTemperature(ctx context.Context, defaultTemperature float64) float64 {
	return ContextValue(ctx, ContextKeyTemperature, defaultTemperature)
}

// WithContextSeed stores a seed value in the context
func WithContextSeed(ctx context.Context, seed int) context.Context {
	return context.WithValue(ctx, ContextKeySeed, seed)
}

// ContextSeed retrieves a seed value from the context
func ContextSeed(ctx context.Context, defaultSeed int) int {
	return ContextValue(ctx, ContextKeySeed, defaultSeed)
}

// WithContextTools stores tools in the context
func WithContextTools(ctx context.Context, tools []llm.Tool) context.Context {
	return context.WithValue(ctx, ContextKeyTools, tools)
}

// ContextTools retrieves tools from the context
func ContextTools(ctx context.Context, defaultTools []llm.Tool) []llm.Tool {
	return ContextValue(ctx, ContextKeyTools, defaultTools)
}

// WithContextMessages stores messages in the context
func WithContextMessages(ctx context.Context, messages []llm.Message) context.Context {
	return context.WithValue(ctx, ContextKeyMessages, messages)
}

// ContextMessages retrieves messages from the context
func ContextMessages(ctx context.Context, defaultMessages []llm.Message) []llm.Message {
	return ContextValue(ctx, ContextKeyMessages, defaultMessages)
}

// ContextValue retrieves a typed value from the context
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

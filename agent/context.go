package agent

import (
	"context"

	"github.com/bornholm/genai/llm"
)

type contextKey string

const (
	contextKeyAgent    contextKey = "agent"
	contextKeyClient   contextKey = "client"
	contextKeyTools    contextKey = "tools"
	contextKeyMessages contextKey = "messages"
)

func WithContextClient(ctx context.Context, client llm.ChatCompletionClient) context.Context {
	return context.WithValue(ctx, contextKeyClient, client)
}

func ContextClient(ctx context.Context, defaultClient llm.ChatCompletionClient) llm.ChatCompletionClient {
	return ContextValue(ctx, contextKeyClient, defaultClient)
}

func WithContextTools(ctx context.Context, tools []llm.Tool) context.Context {
	return context.WithValue(ctx, contextKeyTools, tools)
}

func ContextTools(ctx context.Context, defaultTools []llm.Tool) []llm.Tool {
	return ContextValue(ctx, contextKeyTools, defaultTools)
}

func WithContextMessages(ctx context.Context, messages []llm.Message) context.Context {
	return context.WithValue(ctx, contextKeyMessages, messages)
}

func ContextMessages(ctx context.Context, defaultMessages []llm.Message) []llm.Message {
	return ContextValue(ctx, contextKeyMessages, defaultMessages)
}

func WithContextAgent(ctx context.Context, agent *Agent) context.Context {
	return context.WithValue(ctx, contextKeyAgent, agent)
}

func ContextAgent(ctx context.Context) *Agent {
	return ContextValue[contextKey, *Agent](ctx, contextKeyAgent, nil)
}

func ContextValue[K any, T any](ctx context.Context, key K, defaultValue T) T {
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

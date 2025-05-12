package simple

import (
	"context"

	"github.com/bornholm/genai/llm"
)

type contextKey string

const (
	contextKeyClient   contextKey = "client"
	contextKeyTools    contextKey = "tools"
	contextKeyMessages contextKey = "messages"
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

func WithContextMessages(ctx context.Context, messages []llm.Message) context.Context {
	return context.WithValue(ctx, contextKeyMessages, messages)
}

func ContextMessages(ctx context.Context, defaultMessages []llm.Message) []llm.Message {
	return contextValue(ctx, contextKeyMessages, defaultMessages)
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

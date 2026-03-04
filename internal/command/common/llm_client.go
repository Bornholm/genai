package common

import (
	"context"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/env"
	"github.com/bornholm/genai/llm/ratelimit"
	"github.com/bornholm/genai/llm/retry"
	"github.com/bornholm/genai/llm/tokenlimit"
	"github.com/pkg/errors"
)

// TokenLimitOptions holds token rate limiting configuration.
type TokenLimitOptions struct {
	// ChatCompletionTokens is the maximum tokens per interval for chat completion.
	ChatCompletionTokens int
	// ChatCompletionInterval is the time interval for chat completion token limit.
	ChatCompletionInterval time.Duration
	// EmbeddingsTokens is the maximum tokens per interval for embeddings.
	EmbeddingsTokens int
	// EmbeddingsInterval is the time interval for embeddings token limit.
	EmbeddingsInterval time.Duration
}

// DefaultTokenLimitOptions returns the default token limit options.
func DefaultTokenLimitOptions() *TokenLimitOptions {
	return &TokenLimitOptions{
		ChatCompletionTokens:   500000,
		ChatCompletionInterval: time.Minute,
		EmbeddingsTokens:       20000000,
		EmbeddingsInterval:     time.Minute,
	}
}

// NewResilientClient creates a new resilient LLM client with retry, rate limiting,
// and optional token limiting.
func NewResilientClient(ctx context.Context, envPrefix string, envFile string, tokenLimitOpts *TokenLimitOptions) (llm.Client, error) {
	var (
		client llm.Client
		err    error
	)

	client, err = provider.Create(ctx, env.With(envPrefix, envFile))
	if err != nil {
		return nil, errors.WithStack(err)
	}

	client = retry.NewClient(client, time.Second*2, 5)
	client = ratelimit.NewClient(client, time.Second*2, 1)

	// Apply token limiting if configured
	if tokenLimitOpts != nil {
		client = tokenlimit.NewClient(client,
			tokenlimit.WithChatCompletionLimit(tokenLimitOpts.ChatCompletionTokens, tokenLimitOpts.ChatCompletionInterval),
			tokenlimit.WithEmbeddingsLimit(tokenLimitOpts.EmbeddingsTokens, tokenLimitOpts.EmbeddingsInterval),
		)
	}

	return client, nil
}

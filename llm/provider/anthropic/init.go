// Package anthropic implements the genai chat completion client for the
// Anthropic Messages API, on top of the official Go SDK.
//
// Only chat completion is registered: Anthropic exposes no embeddings,
// transcription or image generation endpoint.
package anthropic

import (
	"context"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

const Name provider.Name = "anthropic"

func init() {
	provider.RegisterChatCompletion(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.ChatCompletionClient, error) {
			options := []option.RequestOption{
				option.WithBaseURL(normalizeBaseURL(opts.BaseURL)),
				option.WithMaxRetries(0), // genai's llmretry wrapper handles all retries
			}
			if opts.APIKey != "" {
				options = append(options, option.WithAPIKey(opts.APIKey))
			}
			client := anthropicsdk.NewClient(options...)

			maxTokens := opts.MaxTokens
			if maxTokens <= 0 {
				maxTokens = DefaultMaxTokens
			}

			return NewChatCompletionClient(client, opts.Model, maxTokens), nil
		},
	)
}

// normalizeBaseURL strips a trailing "/v1" from the configured base URL: the
// SDK appends "v1/messages" itself, and operators used to OpenAI-compatible
// providers routinely configure "https://api.anthropic.com/v1". An empty
// value falls back to the production API: the registry hands Options built
// by the caller straight to the factory, without merging the defaults.
func normalizeBaseURL(baseURL string) string {
	trimmed := strings.TrimRight(baseURL, "/")
	trimmed = strings.TrimSuffix(trimmed, "/v1")
	if trimmed == "" {
		return defaultOptions().BaseURL + "/"
	}
	return trimmed + "/"
}

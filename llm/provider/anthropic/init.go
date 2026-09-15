// Package anthropic implements the genai chat completion client for the
// Anthropic Messages API, on top of the official Go SDK.
//
// Only chat completion is registered: Anthropic exposes no embeddings,
// transcription or image generation endpoint.
//
// Where the Messages API departs from the shared genai options, the
// provider absorbs the difference instead of surfacing a remote 400:
//
//   - max_tokens is mandatory; DefaultMaxTokens (or MAX_TOKENS) applies when
//     the caller sets none.
//   - Temperature ranges over [0, 1]; a higher value is clamped to 1, and
//     any value is dropped when thinking is enabled, as the API then only
//     accepts its default.
//   - System messages are hoisted into the top-level system field wherever
//     they appear in the conversation; a conversation made of system
//     messages only is refused with a validation error.
//   - Consecutive messages of the same role are folded into one turn, tool
//     results leading a user turn and thinking blocks leading an assistant
//     turn, as the API requires.
//   - The JSON response format requires a schema.
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

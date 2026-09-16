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
//   - The JSON response format uses the API's structured output when a
//     schema is given; without one, a system instruction asks for a single
//     JSON object, as the schema-less OpenAI mode expects the prompt to do.
//     The strict flag of a schema has no effect: the API has no lenient
//     structured mode, the schema is always enforced.
//   - Reasoning from another provider cannot be replayed: it carries no
//     signature. An assistant message made of such reasoning only is
//     refused, one that also carries text is replayed as text alone.
//   - Only the configured API key is used: the SDK's fallbacks on the host's
//     environment, profiles and identity federation are disabled.
//   - When several messages fold into one turn, the cached prefix follows
//     the wire order of the turn (tool results first), not the order of
//     the messages.
//   - In a stream, a tool_use block opened with a non-empty input is taken
//     as complete and its later input deltas are ignored; the API itself
//     always opens the block empty, this only concerns gateways that ship
//     the whole input at once.
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
			// The SDK would otherwise fall back on the host's ambient
			// identity (ANTHROPIC_API_KEY, auth token, on-disk profiles,
			// workload identity federation). A gateway carrying one account
			// per provider must fail on a missing key, not bill the host.
			client := anthropicsdk.NewClient(
				option.WithoutEnvironmentDefaults(),
				option.WithBaseURL(normalizeBaseURL(opts.BaseURL)),
				option.WithMaxRetries(0), // genai's llmretry wrapper handles all retries
				option.WithAPIKey(opts.APIKey),
			)

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

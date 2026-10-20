// Package gemini implements the genai chat completion client for the Gemini
// API (models/{model}:generateContent, and its streamGenerateContent
// variant), as served by Google AI Studio and by the gateways that expose it.
//
// Only chat completion and model discovery are registered.
//
// Where the Gemini API departs from the shared genai options, the provider
// absorbs the difference:
//
//   - System messages are hoisted into the system instruction, wherever they
//     appear. Consecutive tool results share one turn, which keeps the
//     answers to parallel function calls together.
//   - Tool schemas and the response schema are sent as plain JSON schemas
//     (parametersJsonSchema, responseJsonSchema), not as Google's OpenAPI
//     subset.
//   - A function response names its function, which a genai tool message
//     does not: the name is taken from the earlier tool call with the same
//     ID, and a result answering no earlier call is refused. A function
//     call the API returns without an ID gets a generated one.
//   - Reasoning is a thinking budget: an explicit budget is sent as is, an
//     effort level becomes a share of max_output_tokens, or a fixed budget
//     when none is set, bounded to [512, 24576]. Gemini 3 still accepts
//     budgets in place of its thinking levels. Disabled reasoning is a zero
//     budget, which 2.5 Pro and Gemini 3 refuse. Thoughts are asked for
//     whenever reasoning is, and withheld from the caller when excluded.
//   - Thought signatures come back as reasoning details (format
//     "google-gemini-v1", keyed by function call ID, an empty ID for a text
//     part) and are replayed on the next turn. The first function call of
//     a turn without one, from another provider, gets the placeholder Google
//     documents for that case; the signatures of thoughts are not kept, as
//     thoughts are not replayed.
//   - The cache hints and SessionID are dropped: the API caches prompt
//     prefixes on its own.
//   - A prompt or an answer blocked by the safety filters is an error,
//     unless the answer had produced some text before.
package gemini

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

const Name provider.Name = "gemini"

func init() {
	provider.RegisterChatCompletion(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.ChatCompletionClient, error) {
			return NewChatCompletionClient(opts.HTTPClient(nil), baseURL(opts), opts.APIKey, opts.Model), nil
		},
	)

	provider.RegisterDiscovery(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.DiscoveryClient, error) {
			return NewDiscoveryClient(opts.HTTPClient(nil), baseURL(opts), opts.APIKey), nil
		},
	)
}

// baseURL falls back to the production API on an empty value: the registry
// hands Options built by the caller straight to the factory, without merging
// the defaults.
func baseURL(opts *Options) string {
	if opts.BaseURL == "" {
		return defaultOptions().BaseURL
	}
	return opts.BaseURL
}

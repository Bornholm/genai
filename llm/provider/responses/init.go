// Package responses implements the genai chat completion client for the
// OpenAI Responses API (POST /responses), which some models are only served
// through, and which OpenAI-compatible gateways expose next to Chat
// Completions.
//
// The wire format is spoken by hand: the pinned openai-go SDK predates the
// encrypted reasoning items this client relies on.
//
// Where the Responses API departs from the shared genai options, the
// provider absorbs the difference:
//
//   - Requests are not stored (store: false) and ask for the encrypted
//     content of reasoning items, which come back as reasoning details
//     (format "openai-responses-v1") and are replayed on the next turn.
//     Reasoning from another provider cannot be replayed and is dropped.
//   - Reasoning takes an effort level only. A token budget is mapped to the
//     effort whose share of max_output_tokens it is closest to, or by
//     absolute thresholds when no max_output_tokens is set. The readable
//     reasoning is the summary the API returns, or the raw reasoning text of
//     the models that return it; Exclude stops asking for a summary.
//   - Tools are sent in non-strict mode: strict mode requires every
//     property to be required, which genai's tool schemas do not promise.
//   - The seed and cache hints are dropped: the API has neither, prompt
//     prefixes are cached on their own. SessionID becomes prompt_cache_key.
//   - A refusal is returned as the answer's content.
//   - An incomplete response (length limit, content filter) returns what
//     was produced, like a chat completion cut by its length.
//
// Embeddings and model discovery go through the OpenAI client: the API
// root serving /responses serves /embeddings and /models too.
package responses

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/openai"
	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const Name provider.Name = "openai-responses"

func init() {
	provider.RegisterChatCompletion(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.ChatCompletionClient, error) {
			return NewChatCompletionClient(nil, baseURL(opts), opts.APIKey, opts.Model), nil
		},
	)

	provider.RegisterEmbeddings(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.EmbeddingsClient, error) {
			return openai.NewEmbeddingsClient(newSDKClient(opts), opts.Model), nil
		},
	)

	provider.RegisterDiscovery(
		Name,
		defaultOptions,
		func(ctx context.Context, opts *Options) (llm.DiscoveryClient, error) {
			return openai.NewDiscoveryClient(newSDKClient(opts)), nil
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

func newSDKClient(opts *Options) openaisdk.Client {
	options := []option.RequestOption{
		option.WithBaseURL(baseURL(opts)),
		option.WithMaxRetries(0), // genai's llmretry wrapper handles all retries
	}
	if opts.APIKey != "" {
		options = append(options, option.WithAPIKey(opts.APIKey))
	}
	return openaisdk.NewClient(options...)
}

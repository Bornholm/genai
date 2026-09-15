package anthropic

import "github.com/bornholm/genai/llm/provider"

// DefaultMaxTokens is the max_tokens sent when the caller does not set
// llm.WithMaxCompletionTokens. The Messages API makes the field mandatory,
// unlike the OpenAI one where it is optional.
const DefaultMaxTokens int64 = 4096

// Options contient les options de configuration du provider Anthropic.
type Options struct {
	provider.CommonOptions

	// MaxTokens is the max_tokens used when a request does not set one.
	MaxTokens int64 `env:"MAX_TOKENS"`
}

func defaultOptions() *Options {
	return &Options{
		CommonOptions: provider.CommonOptions{
			BaseURL: "https://api.anthropic.com",
		},
		MaxTokens: DefaultMaxTokens,
	}
}

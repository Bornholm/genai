package openrouter

import "github.com/bornholm/genai/llm/provider"

// Options contient les options de configuration du provider OpenRouter.
type Options struct {
	provider.CommonOptions
}

func defaultOptions() *Options {
	return &Options{
		CommonOptions: provider.CommonOptions{
			BaseURL: "https://openrouter.ai/api/v1",
		},
	}
}

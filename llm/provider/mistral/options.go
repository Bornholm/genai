package mistral

import "github.com/bornholm/genai/llm/provider"

// Options contient les options de configuration du provider Mistral.
type Options struct {
	provider.CommonOptions
}

func defaultOptions() *Options {
	return &Options{
		CommonOptions: provider.CommonOptions{
			BaseURL: "https://api.mistral.ai/v1",
		},
	}
}

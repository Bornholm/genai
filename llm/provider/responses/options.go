package responses

import "github.com/bornholm/genai/llm/provider"

// Options holds the Responses provider configuration.
type Options struct {
	provider.CommonOptions
}

func defaultOptions() *Options {
	return &Options{
		CommonOptions: provider.CommonOptions{
			BaseURL: "https://api.openai.com/v1",
		},
	}
}

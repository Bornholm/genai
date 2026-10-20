package gemini

import "github.com/bornholm/genai/llm/provider"

// Options holds the Gemini provider configuration.
type Options struct {
	provider.CommonOptions
}

func defaultOptions() *Options {
	return &Options{
		CommonOptions: provider.CommonOptions{
			BaseURL: "https://generativelanguage.googleapis.com/v1beta",
		},
	}
}

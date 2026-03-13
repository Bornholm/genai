package openai

import "github.com/bornholm/genai/llm/provider"

// Options contient les options de configuration du provider OpenAI.
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

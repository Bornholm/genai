package typesafe

import "github.com/bornholm/genai/llm/provider"

// Options contient les options de configuration du provider TypeSafe.
type Options struct {
	provider.CommonOptions
}

func defaultOptions() *Options {
	return &Options{
		CommonOptions: provider.CommonOptions{
			BaseURL: DefaultBaseURL,
			Model:   DefaultModel,
		},
	}
}

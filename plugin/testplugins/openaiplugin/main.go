// Command openaiplugin wraps the in-tree openai provider in a plugin, so
// that the conformance suite can be run through the plugin protocol.
package main

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/openai"
	"github.com/bornholm/genai/plugin/sdk"
)

func main() {
	sdk.Serve(sdk.Config{
		Name:    "openai-plugin",
		Version: "0.0.1",
		ChatCompletion: func(ctx context.Context, opts sdk.Options) (llm.ChatCompletionClient, error) {
			o, err := decode(opts)
			if err != nil {
				return nil, err
			}
			client, err := provider.Create(ctx, provider.WithChatCompletion(openai.Name, *o))
			if err != nil {
				return nil, err
			}
			return client, nil
		},
		Embeddings: func(ctx context.Context, opts sdk.Options) (llm.EmbeddingsClient, error) {
			o, err := decode(opts)
			if err != nil {
				return nil, err
			}
			client, err := provider.Create(ctx, provider.WithEmbeddings(openai.Name, *o))
			if err != nil {
				return nil, err
			}
			return client, nil
		},
	})
}

func decode(opts sdk.Options) (*openai.Options, error) {
	o := &openai.Options{CommonOptions: provider.CommonOptions{BaseURL: "https://api.openai.com/v1"}}
	if err := opts.Decode(o); err != nil {
		return nil, err
	}
	return o, nil
}

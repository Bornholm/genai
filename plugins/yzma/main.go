// Command genai-provider-yzma serves the yzma provider (llama.cpp bindings)
// as a genai plugin. Build it and put it in the plugin directory the host
// searches, GENAI_PLUGIN_DIR, or name it through
// GENAI_<CAPABILITY>_YZMA_COMMAND. The PATH is not searched.
//
//	GENAI_PLUGIN_DIR=/opt/genai/plugins
//	GENAI_CHAT_COMPLETION_PROVIDER=yzma
//	GENAI_CHAT_COMPLETION_YZMA_MODEL_PATH=/models/qwen2.5-0.5b.gguf
//	GENAI_CHAT_COMPLETION_YZMA_LIB_PATH=/opt/llama.cpp/lib
//	GENAI_EMBEDDINGS_PROVIDER=yzma
//	GENAI_EMBEDDINGS_YZMA_MODEL_PATH=/models/nomic-embed.gguf
package main

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/plugin/sdk"
	yzma "github.com/bornholm/genai/plugins/yzma/provider"
)

var version = "dev"

func main() {
	sdk.Serve(sdk.Config{
		Name:    string(yzma.Name),
		Version: version,
		ChatCompletion: func(ctx context.Context, opts sdk.Options) (llm.ChatCompletionClient, error) {
			o := yzma.DefaultChatCompletionOptions()
			if err := opts.Decode(o); err != nil {
				return nil, err
			}
			return yzma.NewChatCompletionClientFromOptions(ctx, o)
		},
		Embeddings: func(ctx context.Context, opts sdk.Options) (llm.EmbeddingsClient, error) {
			o := yzma.DefaultEmbeddingsOptions()
			if err := opts.Decode(o); err != nil {
				return nil, err
			}
			return yzma.NewEmbeddingsClientFromOptions(ctx, o)
		},
	})
}

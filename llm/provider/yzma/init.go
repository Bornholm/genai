package yzma

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/pkg/errors"
)

const Name provider.Name = "yzma"

func init() {
	provider.RegisterChatCompletion(Name, func(ctx context.Context, opts provider.ClientOptions) (llm.ChatCompletionClient, error) {
		// Extract model path from Model field
		modelPath := opts.Model
		if modelPath == "" {
			return nil, errors.New("model path is required for yzma provider")
		}

		// Create client with options
		client, err := NewChatCompletionClient(
			WithModelPath(modelPath),
			WithLibPath(opts.BaseURL), // Use BaseURL field for lib path
		)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		return client, nil
	})

	provider.RegisterEmbeddings(Name, func(ctx context.Context, opts provider.ClientOptions) (llm.EmbeddingsClient, error) {
		// Extract model path from Model field
		modelPath := opts.Model
		if modelPath == "" {
			return nil, errors.New("model path is required for yzma provider")
		}

		// Create client with options
		client, err := NewEmbeddingsClient(
			WithEmbeddingsModelPath(modelPath),
			WithEmbeddingsLibPath(opts.BaseURL), // Use BaseURL field for lib path
		)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		return client, nil
	})
}

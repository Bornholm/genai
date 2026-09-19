package yzma

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/pkg/errors"
)

const Name provider.Name = "yzma"

func init() {
	provider.RegisterChatCompletion(Name, DefaultChatCompletionOptions, NewChatCompletionClientFromOptions)
	provider.RegisterEmbeddings(Name, DefaultEmbeddingsOptions, NewEmbeddingsClientFromOptions)
}

// NewChatCompletionClientFromOptions builds a chat completion client from a
// populated options struct. It is what the registry and the plugin binary
// both call.
func NewChatCompletionClientFromOptions(ctx context.Context, opts *ChatCompletionOptions) (llm.ChatCompletionClient, error) {
	client, err := NewChatCompletionClient(
		WithModelPath(opts.ModelPath),
		WithModelURL(opts.ModelURL),
		WithLibPath(opts.LibPath),
		WithProcessor(opts.Processor),
		WithVersion(opts.Version),
		WithContextSize(opts.ContextSize),
		WithBatchSize(opts.BatchSize),
		WithUBatchSize(opts.UBatchSize),
		WithTemperature(opts.Temperature),
		WithTopK(opts.TopK),
		WithTopP(opts.TopP),
		WithMinP(opts.MinP),
		WithPresencePenalty(opts.PresencePenalty),
		WithPenaltyLastN(opts.PenaltyLastN),
		WithPredictSize(opts.PredictSize),
		WithTemplate(opts.Template),
		WithVerbose(opts.Verbose),
	)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	return client, nil
}

// NewEmbeddingsClientFromOptions builds an embeddings client from a populated
// options struct.
func NewEmbeddingsClientFromOptions(ctx context.Context, opts *EmbeddingsOptions) (llm.EmbeddingsClient, error) {
	client, err := NewEmbeddingsClient(
		WithEmbeddingsModelPath(opts.ModelPath),
		WithEmbeddingsModelURL(opts.ModelURL),
		WithEmbeddingsLibPath(opts.LibPath),
		WithEmbeddingsProcessor(opts.Processor),
		WithEmbeddingsVersion(opts.Version),
		WithEmbeddingsContextSize(opts.ContextSize),
		WithEmbeddingsBatchSize(opts.BatchSize),
		WithEmbeddingsPoolingType(llama.PoolingType(opts.PoolingType)),
		WithEmbeddingsNormalize(opts.Normalize),
		WithEmbeddingsVerbose(opts.Verbose),
	)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	return client, nil
}

package llm

import "context"

type EmbeddingsClient interface {
	Embeddings(ctx context.Context, input string, funcs ...EmbeddingsOptionFunc) (EmbeddingsResponse, error)
}

type EmbeddingsOptions struct {
	Dimensions *int
}

func NewEmbeddingsOptions(funcs ...EmbeddingsOptionFunc) *EmbeddingsOptions {
	opts := &EmbeddingsOptions{}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

type EmbeddingsOptionFunc func(opts *EmbeddingsOptions)

func WithDimensions(dimensions int) EmbeddingsOptionFunc {
	return func(opts *EmbeddingsOptions) {
		opts.Dimensions = &dimensions
	}
}

type EmbeddingsResponse interface {
	Embeddings() [][]float64
	Usage() EmbeddingsUsage
}

type EmbeddingsUsage interface {
	PromptTokens() int64
	TotalTokens() int64
}

type BaseEmbeddingsUsage struct {
	promptTokens int64
	totalTokens  int64
}

// TotalTokens implements EmbeddingsUsage.
func (u *BaseEmbeddingsUsage) TotalTokens() int64 {
	return u.totalTokens
}

// PromptTokens implements EmbeddingsUsage.
func (u *BaseEmbeddingsUsage) PromptTokens() int64 {
	return u.promptTokens
}

func NewEmbeddingsUsage(promptTokens, totalTokens int64) *BaseEmbeddingsUsage {
	return &BaseEmbeddingsUsage{
		promptTokens: promptTokens,
		totalTokens:  totalTokens,
	}
}

var _ EmbeddingsUsage = &BaseEmbeddingsUsage{}

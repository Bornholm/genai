package llm

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
}

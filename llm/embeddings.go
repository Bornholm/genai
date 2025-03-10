package llm

type EmbeddingsOptions struct {
	Input      string
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

func WithInput(input string) EmbeddingsOptionFunc {
	return func(opts *EmbeddingsOptions) {
		opts.Input = input
	}
}

func WithDimensions(dimensions int) EmbeddingsOptionFunc {
	return func(opts *EmbeddingsOptions) {
		opts.Dimensions = &dimensions
	}
}

type EmbeddingsResponse interface {
	Embeddings() [][]float64
}

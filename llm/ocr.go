package llm

type OCROptions struct {
}

func NewOCROptions(funcs ...OCROptionFunc) *OCROptions {
	opts := &OCROptions{}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

type OCROptionFunc func(opts *OCROptions)

type OCRResponse interface {
}

package llm

import (
	"context"
	"io"
	"os"

	"github.com/pkg/errors"
)

var (
	ErrMissingReader = errors.New("missing reader")
)

type ExtractTextClient interface {
	ExtractText(ctx context.Context, funcs ...ExtractTextOptionFunc) (ExtractTextResponse, error)
}

type ExtractTextFormat string

const (
	ExtractTextFormatMarkdown ExtractTextFormat = "markdown"
	ExtractTextFormatText     ExtractTextFormat = "text"
)

type ExtractTextOptions struct {
	Reader io.Reader
	Pages  []int
}

func NewExtractTextOptions(funcs ...ExtractTextOptionFunc) (*ExtractTextOptions, error) {
	opts := &ExtractTextOptions{}
	for _, fn := range funcs {
		if err := fn(opts); err != nil {
			return nil, errors.WithStack(err)
		}
	}
	return opts, nil
}

type ExtractTextOptionFunc func(opts *ExtractTextOptions) error

type ExtractTextResponse interface {
	Output() io.Reader
	Format() ExtractTextFormat
}

func WithFile(path string) ExtractTextOptionFunc {
	return func(opts *ExtractTextOptions) error {
		file, err := os.Open(path)
		if err != nil {
			return errors.WithStack(err)
		}

		opts.Reader = file

		return nil
	}
}

func WithReader(r io.Reader) ExtractTextOptionFunc {
	return func(opts *ExtractTextOptions) error {
		opts.Reader = r
		return nil
	}
}

func WithPages(pages ...int) ExtractTextOptionFunc {
	return func(opts *ExtractTextOptions) error {
		opts.Pages = pages
		return nil
	}
}

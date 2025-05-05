package extract

import (
	"context"
	"io"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
)

var (
	ErrMissingReader = errors.New("missing reader")
)

type TextClient interface {
	Text(ctx context.Context, funcs ...TextOptionFunc) (TextResponse, error)
}

type TextFormat string

const (
	TextFormatMarkdown TextFormat = "markdown"
	TextFormatPlain    TextFormat = "plain"
)

type TextOptions struct {
	Reader   io.Reader
	Pages    []int
	Filename string
}

func NewTextOptions(funcs ...TextOptionFunc) (*TextOptions, error) {
	opts := &TextOptions{}
	for _, fn := range funcs {
		if err := fn(opts); err != nil {
			return nil, errors.WithStack(err)
		}
	}
	return opts, nil
}

type TextOptionFunc func(opts *TextOptions) error

type TextResponse interface {
	Output() io.Reader
	Format() TextFormat
}

func WithFile(path string) TextOptionFunc {
	return func(opts *TextOptions) error {
		file, err := os.Open(path)
		if err != nil {
			return errors.WithStack(err)
		}

		opts.Filename = filepath.Base(path)
		opts.Reader = file

		return nil
	}
}

func WithReader(r io.Reader) TextOptionFunc {
	return func(opts *TextOptions) error {
		opts.Reader = r
		return nil
	}
}

func WithPages(pages ...int) TextOptionFunc {
	return func(opts *TextOptions) error {
		opts.Pages = pages
		return nil
	}
}

func WithFilename(filename string) TextOptionFunc {
	return func(opts *TextOptions) error {
		opts.Filename = filename
		return nil
	}
}

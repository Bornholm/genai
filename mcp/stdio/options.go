package stdio

import (
	"os"
)

type Options struct {
	Env []string
}

type OptionFunc func(opts *Options)

func NewOptions(funcs ...OptionFunc) *Options {
	opts := &Options{
		Env: os.Environ(),
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

func WithEnv(env ...string) OptionFunc {
	return func(opts *Options) {
		opts.Env = env
	}
}

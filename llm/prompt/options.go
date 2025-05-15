package prompt

import "text/template"

type Options struct {
	Funcs template.FuncMap
}

type OptionFunc func(opts *Options)

func WithFuncs(funcs template.FuncMap) OptionFunc {
	return func(opts *Options) {
		opts.Funcs = funcs
	}
}

func NewOptions(funcs ...OptionFunc) *Options {
	opts := &Options{
		Funcs: template.FuncMap{},
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

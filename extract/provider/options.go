package provider

type Options struct {
	TextDSN string `env:"TEXT_DSN"`
}

func WithTextClientDSN(dsn string) OptionFunc {
	return func(opts *Options) {
		opts.TextDSN = dsn
	}
}

type OptionFunc func(opts *Options)

func NewOptions(funcs ...OptionFunc) (*Options, error) {
	opts := &Options{}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts, nil
}

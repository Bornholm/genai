package agent

import "time"

const (
	defaultInputChanSize              = 100
	defaultOutputChanSize             = 100
	defaultErrorChanSize              = 100
	defaultHandlerOutputChanSize      = 10
	defaultConcurrentHandlers         = 10
	defaultOutputChannelTimeout       = 30 * time.Second
	defaultOutputChannelTicker        = 10 * time.Millisecond
	defaultHandlerContextCancelOnDone = true
	defaultInputTimeout               = 1 * time.Second
)

type Options struct {
	InputChanSize              int
	OutputChanSize             int
	ErrorChanSize              int
	HandlerOutputChanSize      int
	ConcurrentHandlers         int
	OutputChannelTimeout       time.Duration
	OutputChannelTicker        time.Duration
	HandlerContextCancelOnDone bool
	InputTimeout               time.Duration
	Middlewares                []Middleware
}

func (o *Options) setDefaults() {
	if o.InputChanSize == 0 {
		o.InputChanSize = defaultInputChanSize
	}
	if o.OutputChanSize == 0 {
		o.OutputChanSize = defaultOutputChanSize
	}
	if o.ErrorChanSize == 0 {
		o.ErrorChanSize = defaultErrorChanSize
	}
	if o.HandlerOutputChanSize == 0 {
		o.HandlerOutputChanSize = defaultHandlerOutputChanSize
	}
	if o.ConcurrentHandlers == 0 {
		o.ConcurrentHandlers = defaultConcurrentHandlers
	}
	if o.OutputChannelTimeout == 0 {
		o.OutputChannelTimeout = defaultOutputChannelTimeout
	}
	if o.OutputChannelTicker == 0 {
		o.OutputChannelTicker = defaultOutputChannelTicker
	}
	if o.InputTimeout == 0 {
		o.InputTimeout = defaultInputTimeout
	}
}

type Option func(*Options)

func WithInputChanSize(size int) Option {
	return func(o *Options) {
		o.InputChanSize = size
	}
}

func WithOutputChanSize(size int) Option {
	return func(o *Options) {
		o.OutputChanSize = size
	}
}

func WithErrorChanSize(size int) Option {
	return func(o *Options) {
		o.ErrorChanSize = size
	}
}

func WithHandlerOutputChanSize(size int) Option {
	return func(o *Options) {
		o.HandlerOutputChanSize = size
	}
}

func WithConcurrentHandlers(size int) Option {
	return func(o *Options) {
		o.ConcurrentHandlers = size
	}
}

func WithOutputChannelTimeout(timeout time.Duration) Option {
	return func(o *Options) {
		o.OutputChannelTimeout = timeout
	}
}

func WithOutputChannelTicker(ticker time.Duration) Option {
	return func(o *Options) {
		o.OutputChannelTicker = ticker
	}
}

func WithHandlerContextCancelOnDone(cancel bool) Option {
	return func(o *Options) {
		o.HandlerContextCancelOnDone = cancel
	}
}

func WithMiddlewares(middlewares ...Middleware) Option {
	return func(o *Options) {
		o.Middlewares = middlewares
	}
}

func WithInputTimeout(timeout time.Duration) Option {
	return func(o *Options) {
		o.InputTimeout = timeout
	}
}

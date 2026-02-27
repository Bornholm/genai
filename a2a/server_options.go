package a2a

import "time"

// ServerOptions contains configuration for the A2A server
type ServerOptions struct {
	Address      string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
}

// ServerOptionFunc is a function that configures the ServerOptions
type ServerOptionFunc func(*ServerOptions)

// NewServerOptions creates a new ServerOptions with defaults
func NewServerOptions(funcs ...ServerOptionFunc) *ServerOptions {
	opts := &ServerOptions{
		Address:      ":8080",
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second, // Long for SSE
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

// WithAddress sets the server address
func WithAddress(addr string) ServerOptionFunc {
	return func(o *ServerOptions) {
		o.Address = addr
	}
}

// WithReadTimeout sets the read timeout
func WithReadTimeout(d time.Duration) ServerOptionFunc {
	return func(o *ServerOptions) {
		o.ReadTimeout = d
	}
}

// WithWriteTimeout sets the write timeout
func WithWriteTimeout(d time.Duration) ServerOptionFunc {
	return func(o *ServerOptions) {
		o.WriteTimeout = d
	}
}

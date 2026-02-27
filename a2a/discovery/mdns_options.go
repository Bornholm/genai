package discovery

import "time"

const (
	DefaultServiceType = "_a2a._tcp"
	DefaultDomain      = "local."
)

// MDNSOptions contains configuration for mDNS discovery
type MDNSOptions struct {
	ServiceType string
	Domain      string
	Instance    string
	Port        int
	BrowseTime  time.Duration
	TXTRecords  []string // Additional TXT records (e.g., "version=1.0")
}

// MDNSOptionFunc is a function that configures the MDNSOptions
type MDNSOptionFunc func(*MDNSOptions)

// NewMDNSOptions creates a new MDNSOptions with defaults
func NewMDNSOptions(funcs ...MDNSOptionFunc) *MDNSOptions {
	opts := &MDNSOptions{
		ServiceType: DefaultServiceType,
		Domain:      DefaultDomain,
		BrowseTime:  5 * time.Second,
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

// WithServiceType sets the service type
func WithServiceType(st string) MDNSOptionFunc {
	return func(o *MDNSOptions) {
		o.ServiceType = st
	}
}

// WithDomain sets the domain
func WithDomain(d string) MDNSOptionFunc {
	return func(o *MDNSOptions) {
		o.Domain = d
	}
}

// WithInstance sets the instance name
func WithInstance(name string) MDNSOptionFunc {
	return func(o *MDNSOptions) {
		o.Instance = name
	}
}

// WithPort sets the port
func WithPort(port int) MDNSOptionFunc {
	return func(o *MDNSOptions) {
		o.Port = port
	}
}

// WithBrowseTime sets the browse time duration
func WithBrowseTime(d time.Duration) MDNSOptionFunc {
	return func(o *MDNSOptions) {
		o.BrowseTime = d
	}
}

// WithTXTRecords sets additional TXT records
func WithTXTRecords(records ...string) MDNSOptionFunc {
	return func(o *MDNSOptions) {
		o.TXTRecords = records
	}
}

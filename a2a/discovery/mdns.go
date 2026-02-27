package discovery

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"strings"
	"time"

	"github.com/grandcat/zeroconf"
	"github.com/pkg/errors"
)

// Announcer registers this agent on the local network via mDNS.
type Announcer struct {
	server *zeroconf.Server
	opts   *MDNSOptions
}

// NewAnnouncer creates and starts an mDNS announcement.
// Call Shutdown() to stop announcing.
func NewAnnouncer(funcs ...MDNSOptionFunc) (*Announcer, error) {
	opts := NewMDNSOptions(funcs...)

	if opts.Instance == "" {
		return nil, errors.New("instance name is required for mDNS announcement")
	}
	if opts.Port == 0 {
		return nil, errors.New("port is required for mDNS announcement")
	}

	server, err := zeroconf.Register(
		opts.Instance,    // instance name
		opts.ServiceType, // service type (e.g. "_a2a._tcp")
		opts.Domain,      // domain
		opts.Port,        // port
		opts.TXTRecords,  // TXT records
		nil,              // interfaces (nil = all)
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to register mDNS service")
	}

	slog.Info("mDNS: announcing agent",
		"instance", opts.Instance,
		"service", opts.ServiceType,
		"port", opts.Port,
	)

	return &Announcer{server: server, opts: opts}, nil
}

// Shutdown stops the mDNS announcement.
func (a *Announcer) Shutdown() {
	if a.server != nil {
		a.server.Shutdown()
		slog.Info("mDNS: stopped announcing agent", "instance", a.opts.Instance)
	}
}

// Browser discovers A2A agents on the local network via mDNS.
type Browser struct {
	opts     *MDNSOptions
	registry *Registry
}

// NewBrowser creates a new mDNS browser.
func NewBrowser(registry *Registry, funcs ...MDNSOptionFunc) *Browser {
	opts := NewMDNSOptions(funcs...)
	return &Browser{
		opts:     opts,
		registry: registry,
	}
}

// Browse performs a one-shot discovery scan for the configured duration.
func (b *Browser) Browse(ctx context.Context) error {
	resolver, err := zeroconf.NewResolver(nil)
	if err != nil {
		return errors.Wrap(err, "failed to create mDNS resolver")
	}

	entries := make(chan *zeroconf.ServiceEntry)

	go func() {
		for entry := range entries {
			agent := serviceEntryToAgent(entry)
			slog.Info("mDNS: discovered agent",
				"name", agent.Name,
				"url", agent.URL,
			)
			b.registry.Add(agent)
		}
	}()

	browseCtx, cancel := context.WithTimeout(ctx, b.opts.BrowseTime)
	defer cancel()

	if err := resolver.Browse(browseCtx, b.opts.ServiceType, b.opts.Domain, entries); err != nil {
		return errors.Wrap(err, "mDNS browse failed")
	}

	<-browseCtx.Done()
	return nil
}

// BrowseContinuous runs discovery continuously, updating the registry
// as agents appear and disappear. Blocks until context is cancelled.
func (b *Browser) BrowseContinuous(ctx context.Context) error {
	ticker := time.NewTicker(b.opts.BrowseTime + 2*time.Second)
	defer ticker.Stop()

	for {
		if err := b.Browse(ctx); err != nil {
			slog.Warn("mDNS browse cycle failed", "error", err)
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			// Continue browsing
		}
	}
}

func serviceEntryToAgent(entry *zeroconf.ServiceEntry) *DiscoveredAgent {
	host := entry.HostName
	if len(entry.AddrIPv4) > 0 {
		host = entry.AddrIPv4[0].String()
	} else if len(entry.AddrIPv6) > 0 {
		host = entry.AddrIPv6[0].String()
	}

	return &DiscoveredAgent{
		ID:      getTXTValue(entry.Text, "id"),
		Name:    unescapeDNSName(entry.Instance),
		Host:    host,
		Port:    entry.Port,
		URL:     fmt.Sprintf("http://%s", net.JoinHostPort(host, fmt.Sprintf("%d", entry.Port))),
		TXT:     entry.Text,
		FoundAt: time.Now(),
	}
}

// getTXTValue extracts a value from TXT records by key
func getTXTValue(txtRecords []string, key string) string {
	prefix := key + "="
	for _, txt := range txtRecords {
		if strings.HasPrefix(txt, prefix) {
			return strings.TrimPrefix(txt, prefix)
		}
	}
	return ""
}

// unescapeDNSName unescapes DNS-SD escaped service instance names.
// DNS-SD uses backslash escaping for special characters:
// - \. = literal dot
// - \\ = literal backslash
// - \  = literal space (backslash followed by space)
// - \DDD = byte value (3 decimal digits)
func unescapeDNSName(name string) string {
	var result strings.Builder
	result.Grow(len(name))

	for i := 0; i < len(name); i++ {
		if name[i] == '\\' && i+1 < len(name) {
			// Check for escaped special characters
			next := name[i+1]
			switch next {
			case '.', ' ', '\\':
				// Escaped special char: \. \  or \\
				result.WriteByte(next)
				i++
				continue
			}
			// Check for \DDD (decimal byte value)
			if i+3 < len(name) && isDigit(name[i+1]) && isDigit(name[i+2]) && isDigit(name[i+3]) {
				// Parse 3-digit decimal value
				val := (int(name[i+1]-'0') * 100) + (int(name[i+2]-'0') * 10) + int(name[i+3]-'0')
				result.WriteByte(byte(val))
				i += 3
				continue
			}
		}
		result.WriteByte(name[i])
	}

	return result.String()
}

func isDigit(c byte) bool {
	return c >= '0' && c <= '9'
}

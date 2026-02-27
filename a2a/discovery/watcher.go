package discovery

import (
	"context"
	"log/slog"
	"time"

	"github.com/grandcat/zeroconf"
	"github.com/pkg/errors"
)

// AgentEventHandler is called when agents are discovered or removed
type AgentEventHandler interface {
	// OnAgentDiscovered is called when a new agent is discovered
	OnAgentDiscovered(agent *DiscoveredAgent)
	// OnAgentRemoved is called when an agent is no longer available
	OnAgentRemoved(agent *DiscoveredAgent)
}

// AgentEventHandlerFunc is an adapter to use functions as handlers
type AgentEventHandlerFunc struct {
	OnDiscovered func(*DiscoveredAgent)
	OnRemoved    func(*DiscoveredAgent)
}

// OnAgentDiscovered implements AgentEventHandler
func (h *AgentEventHandlerFunc) OnAgentDiscovered(agent *DiscoveredAgent) {
	if h.OnDiscovered != nil {
		h.OnDiscovered(agent)
	}
}

// OnAgentRemoved implements AgentEventHandler
func (h *AgentEventHandlerFunc) OnAgentRemoved(agent *DiscoveredAgent) {
	if h.OnRemoved != nil {
		h.OnRemoved(agent)
	}
}

var _ AgentEventHandler = &AgentEventHandlerFunc{}

// Watcher continuously discovers A2A agents and notifies handlers of changes
type Watcher struct {
	opts     *MDNSOptions
	registry *Registry
	handler  AgentEventHandler
}

// NewWatcher creates a new agent watcher
func NewWatcher(handler AgentEventHandler, funcs ...MDNSOptionFunc) *Watcher {
	opts := NewMDNSOptions(funcs...)
	return &Watcher{
		opts:     opts,
		registry: NewRegistry(),
		handler:  handler,
	}
}

// Watch starts continuous discovery and notifies the handler of changes.
// It blocks until the context is cancelled.
func (w *Watcher) Watch(ctx context.Context) error {
	// Use a shorter browse time for more responsive updates
	browseInterval := w.opts.BrowseTime
	if browseInterval < 5*time.Second {
		browseInterval = 5 * time.Second
	}

	ticker := time.NewTicker(browseInterval)
	defer ticker.Stop()

	// Initial scan
	if err := w.scan(ctx); err != nil {
		slog.Warn("initial mDNS scan failed", "error", err)
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := w.scan(ctx); err != nil {
				slog.Warn("mDNS scan failed", "error", err)
			}
		}
	}
}

// scan performs a single discovery scan and detects changes
func (w *Watcher) scan(ctx context.Context) error {
	resolver, err := zeroconf.NewResolver(nil)
	if err != nil {
		return errors.Wrap(err, "failed to create mDNS resolver")
	}

	entries := make(chan *zeroconf.ServiceEntry)

	// Collect discovered agents
	discovered := make(map[string]*DiscoveredAgent)

	go func() {
		for entry := range entries {
			agent := serviceEntryToAgent(entry)
			// Use ID as key if available, otherwise fall back to Name
			key := agent.ID
			if key == "" {
				key = agent.Name
			}
			discovered[key] = agent
		}
	}()

	browseCtx, cancel := context.WithTimeout(ctx, w.opts.BrowseTime)
	defer cancel()

	if err := resolver.Browse(browseCtx, w.opts.ServiceType, w.opts.Domain, entries); err != nil {
		return errors.Wrap(err, "mDNS browse failed")
	}

	<-browseCtx.Done()

	// Detect new agents
	for id, agent := range discovered {
		if _, exists := w.registry.Get(id); !exists {
			w.registry.Add(agent)
			slog.Info("discovered new agent", "name", agent.Name, "id", agent.ID, "url", agent.URL)
			w.handler.OnAgentDiscovered(agent)
		}
	}

	// Detect removed agents
	for _, agent := range w.registry.List() {
		// Use ID as key if available, otherwise fall back to Name
		key := agent.ID
		if key == "" {
			key = agent.Name
		}
		if _, exists := discovered[key]; !exists {
			w.registry.Remove(agent.ID)
			slog.Info("agent removed", "name", agent.Name, "id", agent.ID)
			w.handler.OnAgentRemoved(agent)
		}
	}

	return nil
}

// GetRegistry returns the underlying registry
func (w *Watcher) GetRegistry() *Registry {
	return w.registry
}

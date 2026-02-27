package discovery

import (
	"sync"
	"time"
)

// DiscoveredAgent represents an agent found on the network.
type DiscoveredAgent struct {
	ID      string    `json:"id"` // Unique identifier (UUID) for self-detection
	Name    string    `json:"name"`
	Host    string    `json:"host"`
	Port    int       `json:"port"`
	URL     string    `json:"url"`           // Constructed as http://host:port
	TXT     []string  `json:"txt,omitempty"` // Raw TXT records
	FoundAt time.Time `json:"foundAt"`
}

// Registry is a thread-safe in-memory registry of discovered agents.
type Registry struct {
	mu     sync.RWMutex
	agents map[string]*DiscoveredAgent // keyed by agent ID (UUID)
}

// NewRegistry creates a new Registry
func NewRegistry() *Registry {
	return &Registry{
		agents: make(map[string]*DiscoveredAgent),
	}
}

// Add adds or updates an agent in the registry
func (r *Registry) Add(agent *DiscoveredAgent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.agents[agent.ID] = agent
}

// Remove removes an agent from the registry by ID
func (r *Registry) Remove(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.agents, id)
}

// List returns all discovered agents
func (r *Registry) List() []*DiscoveredAgent {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]*DiscoveredAgent, 0, len(r.agents))
	for _, a := range r.agents {
		result = append(result, a)
	}
	return result
}

// Get retrieves an agent by ID
func (r *Registry) Get(id string) (*DiscoveredAgent, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	a, ok := r.agents[id]
	return a, ok
}

// Count returns the number of agents in the registry
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.agents)
}

// Clear removes all agents from the registry
func (r *Registry) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.agents = make(map[string]*DiscoveredAgent)
}

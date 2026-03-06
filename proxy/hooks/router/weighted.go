package router

import (
	"context"
	"math/rand"
	"sync"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/proxy"
)

// WeightedBackend is a single backend with an associated weight and real model name.
type WeightedBackend struct {
	Client llm.Client
	Weight int    // relative weight (>0); higher = more traffic
	Model  string // real model name at the provider
}

// WeightedRouter performs weighted random selection among backends per model.
type WeightedRouter struct {
	mu       sync.Mutex
	backends map[string][]WeightedBackend // proxy model → backends
	rng      *rand.Rand
	priority int
}

// Name implements proxy.Hook.
func (r *WeightedRouter) Name() string { return "router.weighted" }

// Priority implements proxy.Hook.
func (r *WeightedRouter) Priority() int { return r.priority }

// ResolveModel implements proxy.ModelResolverHook.
func (r *WeightedRouter) ResolveModel(ctx context.Context, req *proxy.ProxyRequest) (llm.Client, string, error) {
	r.mu.Lock()
	backends := r.backends[req.Model]
	r.mu.Unlock()

	if len(backends) == 0 {
		return nil, "", proxy.ErrModelNotFound
	}

	backend := r.pick(backends)
	model := backend.Model
	if model == "" {
		model = req.Model
	}
	return backend.Client, model, nil
}

// ListModels implements proxy.ModelListerHook.
func (r *WeightedRouter) ListModels(ctx context.Context) ([]proxy.ModelInfo, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	models := make([]proxy.ModelInfo, 0, len(r.backends))
	for proxyModel := range r.backends {
		models = append(models, proxy.ModelInfo{
			ID:      proxyModel,
			OwnedBy: "proxy",
		})
	}
	return models, nil
}

// pick selects a backend using weighted random selection.
func (r *WeightedRouter) pick(backends []WeightedBackend) WeightedBackend {
	total := 0
	for _, b := range backends {
		if b.Weight > 0 {
			total += b.Weight
		}
	}
	if total == 0 {
		return backends[0]
	}

	r.mu.Lock()
	n := r.rng.Intn(total)
	r.mu.Unlock()

	cum := 0
	for _, b := range backends {
		if b.Weight <= 0 {
			continue
		}
		cum += b.Weight
		if n < cum {
			return b
		}
	}
	return backends[len(backends)-1]
}

// AddBackend adds a backend for a proxy model at runtime.
func (r *WeightedRouter) AddBackend(proxyModel string, backend WeightedBackend) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.backends[proxyModel] = append(r.backends[proxyModel], backend)
}

// NewWeightedRouter creates a WeightedRouter.
func NewWeightedRouter(backends map[string][]WeightedBackend, priority int) *WeightedRouter {
	r := &WeightedRouter{
		backends: make(map[string][]WeightedBackend, len(backends)),
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
		priority: priority,
	}
	for model, bks := range backends {
		r.backends[model] = append(r.backends[model], bks...)
	}
	return r
}

var _ proxy.ModelResolverHook = &WeightedRouter{}
var _ proxy.ModelListerHook = &WeightedRouter{}

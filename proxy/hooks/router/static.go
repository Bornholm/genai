package router

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/proxy"
)

// StaticRouter routes requests by mapping model names to llm.Client instances.
type StaticRouter struct {
	routes   map[string]routeEntry
	priority int
}

type routeEntry struct {
	client llm.Client
	model  string // real model name at the provider (may differ from proxy name)
}

// Name implements proxy.Hook.
func (r *StaticRouter) Name() string { return "router.static" }

// Priority implements proxy.Hook.
func (r *StaticRouter) Priority() int { return r.priority }

// ResolveModel implements proxy.ModelResolverHook.
func (r *StaticRouter) ResolveModel(ctx context.Context, req *proxy.ProxyRequest) (llm.Client, string, error) {
	entry, ok := r.routes[req.Model]
	if !ok {
		return nil, "", proxy.ErrModelNotFound
	}
	return entry.client, entry.model, nil
}

// ListModels implements proxy.ModelListerHook.
func (r *StaticRouter) ListModels(ctx context.Context) ([]proxy.ModelInfo, error) {
	models := make([]proxy.ModelInfo, 0, len(r.routes))
	for proxyModel := range r.routes {
		models = append(models, proxy.ModelInfo{
			ID:      proxyModel,
			OwnedBy: "proxy",
		})
	}
	return models, nil
}

// NewStaticRouter creates a StaticRouter.
// routes maps proxy model name → (client, real model name).
func NewStaticRouter(routes map[string]Route, priority int) *StaticRouter {
	r := &StaticRouter{
		routes:   make(map[string]routeEntry, len(routes)),
		priority: priority,
	}
	for proxyModel, route := range routes {
		model := route.Model
		if model == "" {
			model = proxyModel
		}
		r.routes[proxyModel] = routeEntry{client: route.Client, model: model}
	}
	return r
}

// Route describes a backend for the static router.
type Route struct {
	Client llm.Client
	Model  string // real model name at the provider; if empty, the proxy model name is used
}

var _ proxy.ModelResolverHook = &StaticRouter{}
var _ proxy.ModelListerHook = &StaticRouter{}

package proxy

import (
	"context"

	"github.com/bornholm/genai/llm"
)

// HookResult tells the chain how to proceed after a hook runs.
type HookResult struct {
	// If non-nil, short-circuits the chain and returns this response.
	Response *ProxyResponse
	// If non-nil, replaces the request for subsequent hooks.
	Request *ProxyRequest
}

// Hook is the central interface for the extensibility system.
type Hook interface {
	// Name returns a unique identifier used for logging/debug.
	Name() string
	// Priority determines execution order (lower = earlier).
	Priority() int
}

// PreRequestHook is called BEFORE the LLM provider call.
// It can modify the request, reject it, or redirect to another backend.
type PreRequestHook interface {
	Hook
	PreRequest(ctx context.Context, req *ProxyRequest) (*HookResult, error)
}

// PostResponseHook is called AFTER receiving the LLM response.
// It can modify the response, log, or account usage.
type PostResponseHook interface {
	Hook
	PostResponse(ctx context.Context, req *ProxyRequest, res *ProxyResponse) (*HookResult, error)
}

// ErrorHook is called when an error occurs during the LLM call.
type ErrorHook interface {
	Hook
	OnError(ctx context.Context, req *ProxyRequest, err error) (*HookResult, error)
}

// ModelResolverHook resolves the requested model to a concrete llm.Client.
// This is the central routing mechanism.
type ModelResolverHook interface {
	Hook
	ResolveModel(ctx context.Context, req *ProxyRequest) (llm.Client, string, error)
}

// ModelInfo describes a model exposed by the proxy.
type ModelInfo struct {
	ID      string
	OwnedBy string
	Created int64
}

// ModelListerHook is an optional extension of ModelResolverHook that can
// enumerate the models it knows about (used by GET /v1/models).
type ModelListerHook interface {
	ModelResolverHook
	ListModels(ctx context.Context) ([]ModelInfo, error)
}

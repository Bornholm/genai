package proxy

import (
	"context"
	"log/slog"
	"sort"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// HookChain executes ordered hook lists for each lifecycle phase.
type HookChain struct {
	preRequest   []PreRequestHook
	postResponse []PostResponseHook
	errorHooks   []ErrorHook
	resolvers    []ModelResolverHook
}

// NewHookChain builds a HookChain from a flat list of hooks, sorting each
// category by Priority() ascending.
func NewHookChain(hooks ...Hook) *HookChain {
	c := &HookChain{}

	for _, h := range hooks {
		if pre, ok := h.(PreRequestHook); ok {
			c.preRequest = append(c.preRequest, pre)
		}
		if post, ok := h.(PostResponseHook); ok {
			c.postResponse = append(c.postResponse, post)
		}
		if errH, ok := h.(ErrorHook); ok {
			c.errorHooks = append(c.errorHooks, errH)
		}
		if res, ok := h.(ModelResolverHook); ok {
			c.resolvers = append(c.resolvers, res)
		}
	}

	sort.Slice(c.preRequest, func(i, j int) bool {
		return c.preRequest[i].Priority() < c.preRequest[j].Priority()
	})
	sort.Slice(c.postResponse, func(i, j int) bool {
		return c.postResponse[i].Priority() < c.postResponse[j].Priority()
	})
	sort.Slice(c.errorHooks, func(i, j int) bool {
		return c.errorHooks[i].Priority() < c.errorHooks[j].Priority()
	})
	sort.Slice(c.resolvers, func(i, j int) bool {
		return c.resolvers[i].Priority() < c.resolvers[j].Priority()
	})

	return c
}

// RunPreRequest iterates pre-request hooks in priority order.
// If any hook returns a non-nil Response, the chain short-circuits.
// If any hook returns a non-nil Request, the modified request propagates.
// Returns (*ProxyResponse, nil) on short-circuit, (nil, nil) to continue normally.
func (c *HookChain) RunPreRequest(ctx context.Context, req *ProxyRequest) (*ProxyResponse, error) {
	for _, h := range c.preRequest {
		result, err := h.PreRequest(ctx, req)
		if err != nil {
			slog.ErrorContext(ctx, "pre-request hook error",
				slog.String("hook", h.Name()),
				slog.Any("error", err),
			)
			return c.RunOnError(ctx, req, errors.WithStack(err))
		}
		if result == nil {
			continue
		}
		if result.Response != nil {
			slog.DebugContext(ctx, "pre-request hook short-circuited",
				slog.String("hook", h.Name()),
			)
			return result.Response, nil
		}
		if result.Request != nil {
			req = result.Request
		}
	}
	return nil, nil
}

// RunPostResponse iterates post-response hooks in priority order.
// Hooks may mutate the response via HookResult.Response.
func (c *HookChain) RunPostResponse(ctx context.Context, req *ProxyRequest, res *ProxyResponse) error {
	for _, h := range c.postResponse {
		result, err := h.PostResponse(ctx, req, res)
		if err != nil {
			slog.ErrorContext(ctx, "post-response hook error",
				slog.String("hook", h.Name()),
				slog.Any("error", err),
			)
			// Non-fatal: log and continue
			continue
		}
		if result != nil && result.Response != nil {
			*res = *result.Response
		}
	}
	return nil
}

// RunOnError iterates error hooks in priority order.
// The first hook that returns a non-nil Response short-circuits and that
// response is returned to the caller.  If no hook handles it, a generic 500
// is returned.
func (c *HookChain) RunOnError(ctx context.Context, req *ProxyRequest, err error) (*ProxyResponse, error) {
	for _, h := range c.errorHooks {
		result, hookErr := h.OnError(ctx, req, err)
		if hookErr != nil {
			slog.ErrorContext(ctx, "error hook itself errored",
				slog.String("hook", h.Name()),
				slog.Any("error", hookErr),
			)
			continue
		}
		if result != nil && result.Response != nil {
			return result.Response, nil
		}
	}

	return nil, nil
}

// ResolveModel tries each resolver in order and returns the first match.
// Returns (nil, "", ErrNoResolver) if no resolver can handle the request.
func (c *HookChain) ResolveModel(ctx context.Context, req *ProxyRequest) (llm.Client, string, error) {
	for _, r := range c.resolvers {
		client, model, err := r.ResolveModel(ctx, req)
		if err != nil {
			if errors.Is(err, ErrModelNotFound) {
				continue
			}
			return nil, "", errors.WithStack(err)
		}
		if client != nil {
			return client, model, nil
		}
	}
	return nil, "", errors.WithStack(ErrNoResolver)
}

// ListModels collects models from all resolvers that implement ModelListerHook.
func (c *HookChain) ListModels(ctx context.Context) ([]ModelInfo, error) {
	var models []ModelInfo
	seen := make(map[string]struct{})

	for _, r := range c.resolvers {
		lister, ok := r.(ModelListerHook)
		if !ok {
			continue
		}
		listed, err := lister.ListModels(ctx)
		if err != nil {
			slog.WarnContext(ctx, "model lister hook failed",
				slog.String("hook", r.Name()),
				slog.Any("error", err),
			)
			continue
		}
		for _, m := range listed {
			if _, dup := seen[m.ID]; dup {
				continue
			}
			seen[m.ID] = struct{}{}
			models = append(models, m)
		}
	}

	return models, nil
}

// ErrModelNotFound is returned by resolvers when they don't handle a model.
var ErrModelNotFound = errors.New("model not found")

// ErrNoResolver is returned when no resolver could handle the model.
var ErrNoResolver = errors.New("no resolver for model")

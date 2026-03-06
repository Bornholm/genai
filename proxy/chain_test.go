package proxy

import (
	"context"
	"net/http"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// ---- helpers ------------------------------------------------------------

type stubPreHook struct {
	hookName     string
	hookPriority int
	result       *HookResult
	err          error
	called       *bool
}

func (h *stubPreHook) Name() string  { return h.hookName }
func (h *stubPreHook) Priority() int { return h.hookPriority }
func (h *stubPreHook) PreRequest(_ context.Context, _ *ProxyRequest) (*HookResult, error) {
	if h.called != nil {
		*h.called = true
	}
	return h.result, h.err
}

type stubPostHook struct {
	hookName     string
	hookPriority int
	called       *bool
}

func (h *stubPostHook) Name() string  { return h.hookName }
func (h *stubPostHook) Priority() int { return h.hookPriority }
func (h *stubPostHook) PostResponse(_ context.Context, _ *ProxyRequest, _ *ProxyResponse) (*HookResult, error) {
	if h.called != nil {
		*h.called = true
	}
	return nil, nil
}

type stubResolver struct {
	hookName     string
	hookPriority int
	client       llm.Client
	model        string
	err          error
}

func (r *stubResolver) Name() string  { return r.hookName }
func (r *stubResolver) Priority() int { return r.hookPriority }
func (r *stubResolver) ResolveModel(_ context.Context, _ *ProxyRequest) (llm.Client, string, error) {
	return r.client, r.model, r.err
}

// ---- tests --------------------------------------------------------------

func TestHookChain_NewHookChain(t *testing.T) {
	chain := NewHookChain()
	if chain == nil {
		t.Fatal("NewHookChain returned nil")
	}
}

func TestHookChain_ShortCircuit(t *testing.T) {
	shortResp := &ProxyResponse{StatusCode: http.StatusForbidden, Body: "blocked"}

	h1Called := false
	h2Called := false

	h1 := &stubPreHook{hookName: "h1", hookPriority: 1, result: &HookResult{Response: shortResp}, called: &h1Called}
	h2 := &stubPreHook{hookName: "h2", hookPriority: 2, called: &h2Called}

	chain := NewHookChain(h1, h2)
	req := &ProxyRequest{Metadata: map[string]any{}}

	resp, err := chain.RunPreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp == nil {
		t.Fatal("expected short-circuit response, got nil")
	}
	if resp.StatusCode != http.StatusForbidden {
		t.Errorf("StatusCode = %d, want %d", resp.StatusCode, http.StatusForbidden)
	}
	if !h1Called {
		t.Error("h1 was not called")
	}
	if h2Called {
		t.Error("h2 should not have been called after short-circuit")
	}
}

func TestHookChain_NoShortCircuit(t *testing.T) {
	h1Called := false
	h2Called := false

	h1 := &stubPreHook{hookName: "h1", hookPriority: 1, called: &h1Called}
	h2 := &stubPreHook{hookName: "h2", hookPriority: 2, called: &h2Called}

	chain := NewHookChain(h1, h2)
	req := &ProxyRequest{Metadata: map[string]any{}}

	resp, err := chain.RunPreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp != nil {
		t.Error("expected no short-circuit")
	}
	if !h1Called || !h2Called {
		t.Error("both hooks should have been called")
	}
}

func TestHookChain_RunOnError_NoHooks_ReturnsNil(t *testing.T) {
	chain := NewHookChain()
	req := &ProxyRequest{Metadata: map[string]any{}}

	resp, err := chain.RunOnError(context.Background(), req, errors.New("boom"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp != nil {
		t.Fatalf("expected nil response when no error hooks are registered, got %+v", resp)
	}
}

func TestHookChain_ResolveModel_NotFound(t *testing.T) {
	chain := NewHookChain()
	req := &ProxyRequest{Model: "gpt-99", Metadata: map[string]any{}}

	_, _, err := chain.ResolveModel(context.Background(), req)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrNoResolver) {
		t.Errorf("expected ErrNoResolver, got %v", err)
	}
}

func TestHookChain_ResolveModel_SkipsNotFound(t *testing.T) {
	r1 := &stubResolver{hookName: "r1", hookPriority: 1, err: ErrModelNotFound}
	r2 := &stubResolver{hookName: "r2", hookPriority: 2, model: "actual-model", client: &mockChatClient{}}

	chain := NewHookChain(r1, r2)
	req := &ProxyRequest{Model: "gpt-4", Metadata: map[string]any{}}

	_, model, err := chain.ResolveModel(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "actual-model" {
		t.Errorf("model = %q, want %q", model, "actual-model")
	}
}

func TestHookChain_PostResponse_Called(t *testing.T) {
	called := false
	h := &stubPostHook{hookName: "h", hookPriority: 1, called: &called}

	chain := NewHookChain(h)
	req := &ProxyRequest{Metadata: map[string]any{}}
	res := &ProxyResponse{StatusCode: http.StatusOK}

	if err := chain.RunPostResponse(context.Background(), req, res); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Error("post-response hook was not called")
	}
}

func TestHookChain_PriorityOrder(t *testing.T) {
	order := []string{}

	makeHook := func(name string, prio int) Hook {
		return &callOrderHook{hookName: name, hookPriority: prio, order: &order}
	}

	chain := NewHookChain(
		makeHook("c", 30),
		makeHook("a", 10),
		makeHook("b", 20),
	)

	req := &ProxyRequest{Metadata: map[string]any{}}
	_, _ = chain.RunPreRequest(context.Background(), req)

	if len(order) != 3 {
		t.Fatalf("expected 3 hooks called, got %d", len(order))
	}
	if order[0] != "a" || order[1] != "b" || order[2] != "c" {
		t.Errorf("wrong order: %v", order)
	}
}

type callOrderHook struct {
	hookName     string
	hookPriority int
	order        *[]string
}

func (h *callOrderHook) Name() string  { return h.hookName }
func (h *callOrderHook) Priority() int { return h.hookPriority }
func (h *callOrderHook) PreRequest(_ context.Context, _ *ProxyRequest) (*HookResult, error) {
	*h.order = append(*h.order, h.hookName)
	return nil, nil
}

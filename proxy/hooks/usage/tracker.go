package usage

import (
	"context"
	"log/slog"
	"time"

	"github.com/bornholm/genai/proxy"
)

// UsageTracker is a PostResponseHook that records token usage for every request.
type UsageTracker struct {
	store    UsageStore
	priority int
}

// Name implements proxy.Hook.
func (t *UsageTracker) Name() string { return "usage.tracker" }

// Priority implements proxy.Hook.
func (t *UsageTracker) Priority() int { return t.priority }

// PostResponse implements proxy.PostResponseHook.
func (t *UsageTracker) PostResponse(ctx context.Context, req *proxy.ProxyRequest, res *proxy.ProxyResponse) (*proxy.HookResult, error) {
	if res.TokensUsed == nil {
		return nil, nil
	}

	record := UsageRecord{
		UserID:           req.UserID,
		Model:            req.Model,
		PromptTokens:     res.TokensUsed.PromptTokens,
		CompletionTokens: res.TokensUsed.CompletionTokens,
		Timestamp:        time.Now(),
		RequestType:      req.Type,
	}

	if err := t.store.Record(ctx, record); err != nil {
		slog.ErrorContext(ctx, "could not record usage",
			slog.String("user", req.UserID),
			slog.Any("error", err),
		)
	}

	return nil, nil
}

// NewUsageTracker creates a UsageTracker backed by the given store.
func NewUsageTracker(store UsageStore, priority int) *UsageTracker {
	return &UsageTracker{store: store, priority: priority}
}

var _ proxy.PostResponseHook = &UsageTracker{}

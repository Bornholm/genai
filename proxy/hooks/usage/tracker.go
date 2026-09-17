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
		TokensKnown:      true,
	}

	// An interrupted stream is recorded like any other — the request was made
	// and the provider billed what it produced — but the counts are only worth
	// what the provider published before it stopped. Saying so is what keeps a
	// quota or a bill from reading a zeroed row as a free request.
	if res.Interruption != nil {
		record.Interrupted = true
		record.TokensKnown = res.Interruption.PartialUsage
		if !record.TokensKnown {
			slog.WarnContext(ctx, "recording an interrupted request with unknown token counts",
				slog.String("user", req.UserID),
				slog.String("model", req.Model),
				slog.String("cause", string(res.Interruption.Cause)),
				slog.Int("chunks_emitted", res.Interruption.ChunksEmitted))
		}
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

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

	// An all-zero row means the provider published nothing, not that the request
	// was free — some report usage only in the final chunk of a stream, and some
	// never ask for it at all. Saying so is what keeps a quota or a bill from
	// reading the row as a free request. Cached tokens and a reported cost count
	// as a measurement, like llm.UsagePublishesCounters upstream.
	record.CachedTokens = res.TokensUsed.CachedTokens
	record.TokensKnown = record.PromptTokens > 0 || record.CompletionTokens > 0 ||
		record.CachedTokens > 0 || (res.TokensUsed.Cost != nil && *res.TokensUsed.Cost > 0)

	// An interrupted stream is recorded like any other — the request was made
	// and the provider billed what it produced — but the counts are only worth
	// what it published before it stopped.
	if res.Interruption != nil {
		record.Interrupted = true
		record.TokensKnown = record.TokensKnown && res.Interruption.PartialUsage
	}

	if !record.TokensKnown {
		msg := "recording a request with unknown token counts"
		attrs := []any{
			slog.String("user", req.UserID),
			slog.String("model", req.Model),
		}
		if res.Interruption != nil {
			attrs = append(attrs,
				slog.String("cause", string(res.Interruption.Cause)),
				slog.Int("chunks_emitted", res.Interruption.ChunksEmitted))
		}
		// Two of these are ordinary traffic and would drown the third if they
		// shared its level: a provider that never reports usage at all leaves
		// every single request unmeasured, and a client hanging up mid-stream is
		// a closed tab. Only an upstream failure or a truncated stream is an
		// incident worth waking someone for.
		switch {
		case res.Interruption == nil,
			res.Interruption.Cause == proxy.StreamInterruptionClientGone:
			slog.DebugContext(ctx, msg, attrs...)
		default:
			slog.WarnContext(ctx, msg, attrs...)
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

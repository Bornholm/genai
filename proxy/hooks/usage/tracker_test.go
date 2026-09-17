package usage

import (
	"context"
	"testing"
	"time"

	"github.com/bornholm/genai/proxy"
)

// TestUsageTracker_MarksInterruptedRecords asserts that a row written for an
// interrupted stream says so, and says whether its counts are real. A zeroed row
// recorded as if it were complete is read downstream as a free request — the
// failure this whole change exists to close.
func TestUsageTracker_MarksInterruptedRecords(t *testing.T) {
	for _, tc := range []struct {
		name            string
		tokens          proxy.TokenUsage
		interruption    *proxy.StreamInterruption
		wantInterrupted bool
		wantKnown       bool
	}{
		{
			name:      "completed stream",
			tokens:    proxy.TokenUsage{PromptTokens: 7, CompletionTokens: 3},
			wantKnown: true,
		},
		{
			// Mistral asks for no usage at all, so a perfectly complete
			// response can carry none: unknown, not free.
			name:   "completed stream, provider published nothing",
			tokens: proxy.TokenUsage{},
		},
		{
			name:   "interrupted with published counters",
			tokens: proxy.TokenUsage{PromptTokens: 7, CompletionTokens: 3},
			interruption: &proxy.StreamInterruption{
				Cause:        proxy.StreamInterruptionUpstream,
				PartialUsage: true,
			},
			wantInterrupted: true,
			wantKnown:       true,
		},
		{
			name:   "interrupted with nothing published",
			tokens: proxy.TokenUsage{PromptTokens: 7, CompletionTokens: 3},
			interruption: &proxy.StreamInterruption{
				Cause: proxy.StreamInterruptionClientGone,
			},
			wantInterrupted: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := NewInMemoryUsageStore()
			tracker := NewUsageTracker(store, 1)

			req := &proxy.ProxyRequest{UserID: "u1", Model: "gpt-4", Type: proxy.RequestTypeChatCompletion}
			tokens := tc.tokens
			res := &proxy.ProxyResponse{
				TokensUsed:   &tokens,
				Interruption: tc.interruption,
			}

			if _, err := tracker.PostResponse(context.Background(), req, res); err != nil {
				t.Fatalf("PostResponse: %v", err)
			}

			records, err := store.GetUsage(context.Background(), "u1", time.Time{})
			if err != nil {
				t.Fatalf("GetUsage: %v", err)
			}
			if len(records) != 1 {
				t.Fatalf("records = %d, want 1", len(records))
			}
			if got := records[0].Interrupted; got != tc.wantInterrupted {
				t.Errorf("Interrupted = %v, want %v", got, tc.wantInterrupted)
			}
			if got := records[0].TokensKnown; got != tc.wantKnown {
				t.Errorf("TokensKnown = %v, want %v", got, tc.wantKnown)
			}
		})
	}
}

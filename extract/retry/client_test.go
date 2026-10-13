package retry

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/bornholm/genai/extract"
)

// rateLimitedClient fails every extraction with a rate limit.
type rateLimitedClient struct{ calls int }

func (c *rateLimitedClient) Text(_ context.Context, _ ...extract.TextOptionFunc) (extract.TextResponse, error) {
	c.calls++
	return nil, extract.ErrRateLimit
}

// TestText_StopsWaitingWhenTheContextEnds asserts that the backoff between two
// attempts gives up as soon as the caller does.
func TestText_StopsWaitingWhenTheContextEnds(t *testing.T) {
	client := &rateLimitedClient{}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := NewClient(client, time.Hour, 3).Text(ctx)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want the context deadline", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("returned after %s, want it to stop waiting with the context", elapsed)
	}
	if client.calls != 1 {
		t.Errorf("calls = %d, want 1", client.calls)
	}
}

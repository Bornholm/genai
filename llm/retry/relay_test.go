package retry

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/bornholm/genai/llm"
)

// relayOpener fails to open its first attempts with the given errors, then
// opens an empty stream.
type relayOpener struct {
	llm.Client
	failures []error
	attempts int
}

func (c *relayOpener) RelayMessages(context.Context, []byte, http.Header) (<-chan llm.StreamChunk, error) {
	c.attempts++
	if c.attempts <= len(c.failures) {
		return nil, c.failures[c.attempts-1]
	}
	ch := make(chan llm.StreamChunk)
	close(ch)
	return ch, nil
}

func TestRelayMessages_RetriesOpeningOnly(t *testing.T) {
	inner := &relayOpener{failures: []error{llm.RateLimitError(http.StatusTooManyRequests, "slow down")}}
	client := NewClient(inner, 0, 2)

	if !llm.SupportsMessagesRelay(client) {
		t.Fatal("the retrying client must pass the relay on")
	}
	if _, err := client.RelayMessages(context.Background(), nil, nil); err != nil {
		t.Fatal(err)
	}
	if inner.attempts != 2 {
		t.Errorf("attempts = %d, want a retry after the 429", inner.attempts)
	}
}

func TestRelayMessages_DoesNotRetryARejectedRequest(t *testing.T) {
	inner := &relayOpener{failures: []error{llm.RateLimitError(http.StatusBadRequest, "bad")}}

	_, err := NewClient(inner, 0, 2).RelayMessages(context.Background(), nil, nil)

	var httpErr *llm.HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusBadRequest || inner.attempts != 1 {
		t.Errorf("err = %v after %d attempts, want the 400 after one", err, inner.attempts)
	}
}

func TestSupportsMessagesRelay_HiddenWithoutARelayingClient(t *testing.T) {
	if llm.SupportsMessagesRelay(NewClient(&scriptedStreamClient{}, 0, 1)) {
		t.Error("a retrying client over a non-relaying one must not claim the relay")
	}
}

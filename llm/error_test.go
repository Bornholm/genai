package llm

import (
	"errors"
	"net/http"
	"testing"
)

func TestRateLimitError(t *testing.T) {
	t.Run("429 wraps ErrRateLimit and exposes HTTPError", func(t *testing.T) {
		err := RateLimitError(http.StatusTooManyRequests, "slow down")

		if !errors.Is(err, ErrRateLimit) {
			t.Fatalf("expected errors.Is(err, ErrRateLimit) to be true for a 429")
		}

		var httpErr *HTTPError
		if !errors.As(err, &httpErr) {
			t.Fatalf("expected the underlying *HTTPError to be retrievable via errors.As")
		}
		if httpErr.StatusCode != http.StatusTooManyRequests {
			t.Errorf("expected StatusCode 429, got %d", httpErr.StatusCode)
		}
		if httpErr.Body != "slow down" {
			t.Errorf("expected body %q, got %q", "slow down", httpErr.Body)
		}
	})

	t.Run("non-429 is not a rate limit", func(t *testing.T) {
		err := RateLimitError(http.StatusInternalServerError, "boom")

		if errors.Is(err, ErrRateLimit) {
			t.Fatalf("expected a 500 not to satisfy errors.Is(err, ErrRateLimit)")
		}

		var httpErr *HTTPError
		if !errors.As(err, &httpErr) {
			t.Fatalf("expected the underlying *HTTPError to be retrievable via errors.As")
		}
		if httpErr.StatusCode != http.StatusInternalServerError {
			t.Errorf("expected StatusCode 500, got %d", httpErr.StatusCode)
		}
	})
}

func TestNewHTTPErrorStaysPlain(t *testing.T) {
	// NewHTTPError must remain a plain constructor with no hidden rate-limit
	// tagging: a 429 built through it alone is not an ErrRateLimit.
	err := error(NewHTTPError(http.StatusTooManyRequests, "slow down"))

	if errors.Is(err, ErrRateLimit) {
		t.Fatalf("NewHTTPError must not tag ErrRateLimit; only RateLimitError does")
	}
}

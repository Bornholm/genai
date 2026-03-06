package llm

import (
	"errors"
	"fmt"
	"net/http"
)

var (
	ErrUnavailable = errors.New("unavailable")
	ErrNoMessage   = errors.New("no message")
	ErrRateLimit   = errors.New("rate limit")
)

// HTTPError is returned by providers when the upstream API responds with a
// non-2xx HTTP status. Callers (e.g. the proxy) can use errors.As to retrieve
// the status code and propagate it faithfully.
//
// HTTPError satisfies errors.Is(err, ErrRateLimit) when StatusCode is 429, so
// retry logic that checks for ErrRateLimit continues to work without changes.
type HTTPError struct {
	StatusCode int
	Body       string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("http %d: %s", e.StatusCode, e.Body)
}

// NewHTTPError creates an HTTPError for the given status code and body.
func NewHTTPError(statusCode int, body string) *HTTPError {
	return &HTTPError{StatusCode: statusCode, Body: body}
}

// IsRetryable reports whether err should be retried.
// It returns true for ErrRateLimit sentinels and for HTTPError responses with
// status 429 (Too Many Requests).
func IsRetryable(err error) bool {
	if errors.Is(err, ErrRateLimit) {
		return true
	}
	var httpErr *HTTPError
	return errors.As(err, &httpErr) && httpErr.StatusCode == http.StatusTooManyRequests
}

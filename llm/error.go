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
// non-2xx HTTP status. It is a plain data carrier: callers can use errors.As to
// retrieve the status code and body and propagate them faithfully.
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

// RateLimitError builds the error a provider returns for a non-2xx upstream
// response, tagging it as a rate limit when appropriate. The underlying
// *HTTPError is always retrievable via errors.As. When statusCode is 429 (Too
// Many Requests) the returned error also wraps ErrRateLimit, so callers can
// detect it with errors.Is(err, ErrRateLimit). Providers call this explicitly at
// the point they observe the upstream status.
func RateLimitError(statusCode int, body string) error {
	httpErr := NewHTTPError(statusCode, body)
	if statusCode == http.StatusTooManyRequests {
		return errors.Join(ErrRateLimit, httpErr)
	}

	return httpErr
}

// IsRetryable reports whether err should be retried.
// It returns true for ErrRateLimit and ErrNoMessage sentinels, and for HTTPError
// responses with status 429 (Too Many Requests) or 5xx server errors.
func IsRetryable(err error) bool {
	if errors.Is(err, ErrRateLimit) || errors.Is(err, ErrNoMessage) {
		return true
	}
	var httpErr *HTTPError
	if errors.As(err, &httpErr) {
		return httpErr.StatusCode == http.StatusTooManyRequests || httpErr.StatusCode >= http.StatusInternalServerError
	}
	return false
}

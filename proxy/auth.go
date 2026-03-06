package proxy

import (
	"net/http"
	"strings"
)

// AuthExtractor extracts a user identity from an HTTP request.
type AuthExtractor func(r *http.Request) (userID string, err error)

// BearerTokenExtractor extracts the token from the Authorization header
// (Authorization: Bearer <token>).  The token itself is used as the userID.
func BearerTokenExtractor() AuthExtractor {
	return func(r *http.Request) (string, error) {
		auth := r.Header.Get("Authorization")
		if auth == "" {
			return "", nil
		}
		parts := strings.SplitN(auth, " ", 2)
		if len(parts) != 2 || !strings.EqualFold(parts[0], "bearer") {
			return "", nil
		}
		return strings.TrimSpace(parts[1]), nil
	}
}

// HeaderExtractor extracts the user identity from a custom HTTP header
// (e.g. X-User-ID).
func HeaderExtractor(header string) AuthExtractor {
	return func(r *http.Request) (string, error) {
		return r.Header.Get(header), nil
	}
}

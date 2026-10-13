package proxy

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"
)

// AuthExtractor extracts a user identity from an HTTP request.
type AuthExtractor func(r *http.Request) (userID string, err error)

// BearerTokenExtractor identifies the caller by the token of the Authorization
// header (Authorization: Bearer <token>). The userID is the fingerprint of the
// token, see TokenFingerprint, never the token itself: the userID ends up in
// logs and in the usage store.
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
		token := strings.TrimSpace(parts[1])
		if token == "" {
			return "", nil
		}
		return TokenFingerprint(token), nil
	}
}

// TokenFingerprint returns the userID BearerTokenExtractor gives a token, for
// instance to key a per-user quota: "sha256:" followed by the first 16
// hexadecimal digits of the SHA-256 of the token.
func TokenFingerprint(token string) string {
	sum := sha256.Sum256([]byte(token))
	return "sha256:" + hex.EncodeToString(sum[:8])
}

// HeaderExtractor extracts the user identity from a custom HTTP header
// (e.g. X-User-ID).
func HeaderExtractor(header string) AuthExtractor {
	return func(r *http.Request) (string, error) {
		return r.Header.Get(header), nil
	}
}

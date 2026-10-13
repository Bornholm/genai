package proxy

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestBearerTokenExtractor_GivesAFingerprintNotTheToken(t *testing.T) {
	extract := BearerTokenExtractor()

	userID := func(header string) string {
		t.Helper()
		r := httptest.NewRequest("POST", "/chat/completions", nil)
		if header != "" {
			r.Header.Set("Authorization", header)
		}
		id, err := extract(r)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return id
	}

	first := userID("Bearer sk-secret-1")
	if strings.Contains(first, "sk-secret-1") {
		t.Errorf("userID = %q, want the token kept out of it", first)
	}
	if first != TokenFingerprint("sk-secret-1") {
		t.Errorf("userID = %q, want TokenFingerprint of the token", first)
	}
	if !strings.HasPrefix(first, "sha256:") || len(first) != len("sha256:")+16 {
		t.Errorf("userID = %q, want sha256: and 16 hexadecimal digits", first)
	}
	if userID("bearer sk-secret-1") != first {
		t.Error("the same token gave two userIDs")
	}
	if userID("Bearer sk-secret-2") == first {
		t.Error("two tokens gave the same userID")
	}

	for _, header := range []string{"", "Bearer ", "Basic abc"} {
		if id := userID(header); id != "" {
			t.Errorf("userID for %q = %q, want anonymous", header, id)
		}
	}
}

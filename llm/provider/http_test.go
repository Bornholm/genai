package provider_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

func TestHTTPClient_KeepsTheBaseWhenNothingIsConfigured(t *testing.T) {
	if (provider.CommonOptions{}).HTTPClient(nil) != nil {
		t.Error("no header configured must keep the SDK's own client")
	}
	base := &http.Client{}
	if (provider.CommonOptions{}).HTTPClient(base) != base {
		t.Error("no header configured must keep the base client")
	}
}

func TestHTTPClient_AddsTheHeadersAndTheSessionID(t *testing.T) {
	var got []http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = append(got, r.Header.Clone())
	}))
	defer server.Close()

	redirects := func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	client := provider.CommonOptions{
		Headers:       map[string]string{"User-Agent": "xolo/1.0", "X-Team": "core"},
		SessionHeader: "x-opencode-session",
	}.HTTPClient(&http.Client{CheckRedirect: redirects})

	if client.CheckRedirect == nil {
		t.Error("the base client's settings must be kept")
	}

	ctx := llm.ContextWithSessionID(context.Background(), "session-1")
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, server.URL, nil)
	req.Header.Set("User-Agent", "sdk/1.0")
	if _, err := client.Do(req); err != nil {
		t.Fatal(err)
	}
	if req.Header.Get("User-Agent") != "sdk/1.0" {
		t.Error("the caller's request must not be modified")
	}

	req, _ = http.NewRequestWithContext(context.Background(), http.MethodGet, server.URL, nil)
	if _, err := client.Do(req); err != nil {
		t.Fatal(err)
	}

	if len(got) != 2 {
		t.Fatalf("got %d requests", len(got))
	}
	if got[0].Get("User-Agent") != "xolo/1.0" || got[0].Get("X-Team") != "core" || got[0].Get("X-Opencode-Session") != "session-1" {
		t.Errorf("first request headers = %v", got[0])
	}
	if got[1].Get("X-Opencode-Session") != "" || got[1].Get("User-Agent") != "xolo/1.0" {
		t.Errorf("a request without a session ID must carry no session header, got %v", got[1])
	}
}

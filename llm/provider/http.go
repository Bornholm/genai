package provider

import (
	"net/http"

	"github.com/bornholm/genai/llm"
)

// HTTPClient returns base with a transport that adds the configured headers
// to every request, or base itself when none is configured. A nil base
// stands for http.DefaultClient, and is returned as nil when nothing is
// configured, so that a provider can keep its SDK's own client.
//
// The fixed headers replace those the SDK sets, User-Agent included. The
// session header carries the session ID of the request context (see
// llm.ContextWithSessionID), and is left out without one.
func (o CommonOptions) HTTPClient(base *http.Client) *http.Client {
	if len(o.Headers) == 0 && o.SessionHeader == "" {
		return base
	}

	client := http.Client{}
	if base != nil {
		client = *base
	}
	next := client.Transport
	if next == nil {
		next = http.DefaultTransport
	}
	client.Transport = &headerTransport{
		next:          next,
		headers:       o.Headers,
		sessionHeader: o.SessionHeader,
	}
	return &client
}

type headerTransport struct {
	next          http.RoundTripper
	headers       map[string]string
	sessionHeader string
}

// RoundTrip implements http.RoundTripper.
func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// A RoundTripper must not modify the request it is given.
	req = req.Clone(req.Context())
	for name, value := range t.headers {
		req.Header.Set(name, value)
	}
	if t.sessionHeader != "" {
		if id := llm.SessionIDFromContext(req.Context()); id != "" {
			req.Header.Set(t.sessionHeader, id)
		}
	}
	return t.next.RoundTrip(req)
}

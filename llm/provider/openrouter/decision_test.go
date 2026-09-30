package openrouter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

// The decisions endpoint sits beside /api/v1, not under it: deriving it
// from the configured API root must drop the version segment, or every
// call 404s.
func TestNewDecisionClient_Endpoint(t *testing.T) {
	for baseURL, want := range map[string]string{
		"":                                "https://openrouter.ai/api/alpha/decisions",
		"https://openrouter.ai/api/v1":    "https://openrouter.ai/api/alpha/decisions",
		"https://openrouter.ai/api/v1/":   "https://openrouter.ai/api/alpha/decisions",
		"https://gateway.internal/api/v1": "https://gateway.internal/api/alpha/decisions",
	} {
		// The endpoint is unexported in the typesafe package; the URL a
		// request actually carries is what the caller would hit.
		var got string
		client := NewDecisionClient(&http.Client{Transport: recorder(func(r *http.Request) string {
			got = r.URL.String()
			return `{"model":"jev-1.13.0","answers":{},"usage":{}}`
		})}, baseURL, "key", "")

		if _, err := client.Decision(context.Background(), "state", llm.Questions{
			"q": llm.NoulQuestion{Instructions: "Urgent?"},
		}); err != nil {
			t.Fatalf("baseURL %q: Decision() = %v", baseURL, err)
		}

		if got != want {
			t.Errorf("baseURL %q: endpoint = %q, want %q", baseURL, got, want)
		}
	}
}

// An unconfigured model must fall back to the OpenRouter slug, which is
// namespaced ("~typesafe/…") unlike the native alias.
func TestNewDecisionClient_DefaultModel(t *testing.T) {
	var got struct {
		Model string `json:"model"`
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		io.WriteString(w, `{"model":"jev-1.13.0","answers":{},"usage":{}}`)
	}))
	defer server.Close()

	client := NewDecisionClient(&http.Client{Transport: redirect(server.URL)}, "", "key", "")

	if _, err := client.Decision(context.Background(), "state", llm.Questions{
		"q": llm.NoulQuestion{Instructions: "Urgent?"},
	}); err != nil {
		t.Fatalf("Decision() = %v", err)
	}

	if got.Model != DefaultDecisionModel {
		t.Errorf("model = %q, want %q", got.Model, DefaultDecisionModel)
	}
}

type recorder func(r *http.Request) string

func (fn recorder) RoundTrip(r *http.Request) (*http.Response, error) {
	body := fn(r)
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}, nil
}

// redirect sends every request to the test server, whatever host the client
// derived, so the default endpoint can be exercised offline.
type redirect string

func (target redirect) RoundTrip(r *http.Request) (*http.Response, error) {
	parsed := *r
	url := *r.URL
	base := strings.TrimPrefix(string(target), "http://")
	url.Scheme = "http"
	url.Host = base
	parsed.URL = &url
	return http.DefaultTransport.RoundTrip(&parsed)
}

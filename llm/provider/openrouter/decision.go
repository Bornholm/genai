package openrouter

import (
	"net/http"
	"strings"

	"github.com/bornholm/genai/llm/provider/typesafe"
)

// decisionEndpoint is OpenRouter's dedicated entry point for typed
// questions (POST /api/alpha/decisions). It sits beside /api/v1, not under
// it, and speaks the TypeSafe payload verbatim — hence the reuse of the
// typesafe client rather than a second implementation.
const decisionEndpoint = "https://openrouter.ai/api/alpha/decisions"

// DefaultDecisionModel is the OpenRouter slug of TypeSafe's flagship model.
const DefaultDecisionModel = "~typesafe/jev-latest"

// NewDecisionClient builds the client. An empty baseURL means the public
// service; otherwise it must point at the API root (".../api/v1"), whose
// version segment is dropped since the decisions endpoint lives one level
// up.
func NewDecisionClient(httpClient *http.Client, baseURL, apiKey, model string) *typesafe.DecisionClient {
	endpoint := decisionEndpoint
	if baseURL != "" {
		root := strings.TrimSuffix(strings.TrimSuffix(baseURL, "/"), "/v1")
		endpoint = strings.TrimSuffix(root, "/") + "/alpha/decisions"
	}

	if model == "" {
		model = DefaultDecisionModel
	}

	return typesafe.NewDecisionClient(endpoint, apiKey, model, typesafe.WithHTTPClient(httpClient))
}

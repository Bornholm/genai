package gemini

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// discoveryPageSize is the largest page the models API serves.
const discoveryPageSize = 1000

// DiscoveryClient lists models through the Gemini models API (GET /models),
// following its pages.
type DiscoveryClient struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

type modelList struct {
	Models []struct {
		Name             string `json:"name"`
		DisplayName      string `json:"displayName"`
		InputTokenLimit  int64  `json:"inputTokenLimit"`
		OutputTokenLimit int64  `json:"outputTokenLimit"`
	} `json:"models"`
	NextPageToken string `json:"nextPageToken"`
}

// Models implements [llm.DiscoveryClient]. The IDs are the bare model IDs
// ("gemini-2.5-flash"), without the "models/" prefix of their resource name.
func (c *DiscoveryClient) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	models := make([]llm.ModelInfo, 0)
	pageToken := ""
	for {
		page, err := c.page(ctx, pageToken)
		if err != nil {
			return nil, errors.WithStack(err)
		}
		for _, m := range page.Models {
			models = append(models, llm.NewModelInfo(strings.TrimPrefix(m.Name, "models/"),
				llm.WithModelDisplayName(m.DisplayName),
				llm.WithModelContextWindow(m.InputTokenLimit),
				llm.WithModelMaxOutputTokens(m.OutputTokenLimit),
			))
		}
		if page.NextPageToken == "" || page.NextPageToken == pageToken {
			return models, nil
		}
		pageToken = page.NextPageToken
	}
}

func (c *DiscoveryClient) page(ctx context.Context, pageToken string) (*modelList, error) {
	query := url.Values{"pageSize": {strconv.Itoa(discoveryPageSize)}}
	if pageToken != "" {
		query.Set("pageToken", pageToken)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/models?"+query.Encode(), nil)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	if c.apiKey != "" {
		req.Header.Set("x-goog-api-key", c.apiKey)
	}

	res, err := c.httpClient.Do(req)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	defer res.Body.Close()

	if res.StatusCode < 200 || res.StatusCode > 299 {
		raw, _ := io.ReadAll(io.LimitReader(res.Body, maxErrorBody))
		return nil, errors.WithStack(llm.RateLimitError(res.StatusCode, string(raw)))
	}

	var page modelList
	if err := json.NewDecoder(res.Body).Decode(&page); err != nil {
		return nil, errors.WithStack(err)
	}
	return &page, nil
}

// NewDiscoveryClient lists the models served at baseURL, the API root
// (".../v1beta"). A nil httpClient uses http.DefaultClient.
func NewDiscoveryClient(httpClient *http.Client, baseURL, apiKey string) *DiscoveryClient {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &DiscoveryClient{
		httpClient: httpClient,
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
	}
}

var _ llm.DiscoveryClient = &DiscoveryClient{}

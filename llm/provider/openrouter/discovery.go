package openrouter

import (
	"context"
	"time"

	"github.com/pkg/errors"
	"github.com/revrost/go-openrouter"

	"github.com/bornholm/genai/llm"
)

// DiscoveryClient lists models through the OpenRouter models API
// (GET /models). It returns the whole catalog, not only the models the
// account preferences allow.
type DiscoveryClient struct {
	client *openrouter.Client
}

// Models implements [llm.DiscoveryClient].
func (c *DiscoveryClient) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	list, err := c.client.ListModels(ctx)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	models := make([]llm.ModelInfo, 0, len(list))
	for _, m := range list {
		funcs := []llm.ModelInfoOptionFunc{
			llm.WithModelDisplayName(m.Name),
		}
		if m.Created > 0 {
			funcs = append(funcs, llm.WithModelCreatedAt(time.Unix(m.Created, 0).UTC()))
		}
		if m.ContextLength != nil {
			funcs = append(funcs, llm.WithModelContextWindow(*m.ContextLength))
		}
		if m.TopProvider.MaxCompletionTokens != nil {
			funcs = append(funcs, llm.WithModelMaxOutputTokens(*m.TopProvider.MaxCompletionTokens))
		}
		models = append(models, llm.NewModelInfo(m.ID, funcs...))
	}

	return models, nil
}

// NewDiscoveryClient lists the models served at baseURL; an empty baseURL
// keeps the SDK's default, the production API.
func NewDiscoveryClient(baseURL, apiKey string) *DiscoveryClient {
	config := openrouter.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}
	return &DiscoveryClient{client: openrouter.NewClientWithConfig(*config)}
}

var _ llm.DiscoveryClient = &DiscoveryClient{}

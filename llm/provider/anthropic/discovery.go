package anthropic

import (
	"context"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/pkg/errors"

	"github.com/bornholm/genai/llm"
)

// discoveryPageSize is the largest page the models API accepts.
const discoveryPageSize = 1000

// DiscoveryClient lists models through the Anthropic models API
// (GET /v1/models), following its pagination.
type DiscoveryClient struct {
	client anthropicsdk.Client
}

// Models implements [llm.DiscoveryClient].
func (c *DiscoveryClient) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	pager := c.client.Models.ListAutoPaging(ctx, anthropicsdk.ModelListParams{
		Limit: anthropicsdk.Int(discoveryPageSize),
	})

	models := make([]llm.ModelInfo, 0)
	for pager.Next() {
		m := pager.Current()
		funcs := []llm.ModelInfoOptionFunc{
			llm.WithModelDisplayName(m.DisplayName),
			llm.WithModelContextWindow(m.MaxInputTokens),
			llm.WithModelMaxOutputTokens(m.MaxTokens),
		}
		// The API sends an epoch value when the release date is unknown.
		if m.CreatedAt.Unix() > 0 {
			funcs = append(funcs, llm.WithModelCreatedAt(m.CreatedAt))
		}
		models = append(models, llm.NewModelInfo(m.ID, funcs...))
	}
	if err := pager.Err(); err != nil {
		return nil, errors.WithStack(err)
	}

	return models, nil
}

func NewDiscoveryClient(client anthropicsdk.Client) *DiscoveryClient {
	return &DiscoveryClient{client: client}
}

var _ llm.DiscoveryClient = &DiscoveryClient{}

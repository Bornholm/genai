package openai

import (
	"context"
	"time"

	"github.com/pkg/errors"

	"github.com/bornholm/genai/llm"

	openaisdk "github.com/openai/openai-go"
)

// DiscoveryClient lists models through the OpenAI models API (GET /models).
// Any OpenAI-compatible gateway serving that route works as well, as long as
// it returns its whole list at once: the route is not paginated, so no
// cursor is ever sent and a gateway capping the page loses the rest.
type DiscoveryClient struct {
	client openaisdk.Client
}

// Models implements [llm.DiscoveryClient].
func (c *DiscoveryClient) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	pager := c.client.Models.ListAutoPaging(ctx)

	models := make([]llm.ModelInfo, 0)
	for pager.Next() {
		m := pager.Current()
		var funcs []llm.ModelInfoOptionFunc
		if m.Created > 0 {
			funcs = append(funcs, llm.WithModelCreatedAt(time.Unix(m.Created, 0).UTC()))
		}
		models = append(models, llm.NewModelInfo(m.ID, funcs...))
	}
	if err := pager.Err(); err != nil {
		return nil, errors.WithStack(err)
	}

	return models, nil
}

func NewDiscoveryClient(client openaisdk.Client) *DiscoveryClient {
	return &DiscoveryClient{client: client}
}

var _ llm.DiscoveryClient = &DiscoveryClient{}

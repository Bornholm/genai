package openrouter

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"github.com/revrost/go-openrouter"
)

type EmbeddingsClient struct {
	client *openrouter.Client
	model  string
}

// Embeddings implements [llm.EmbeddingsClient].
func (c *EmbeddingsClient) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	if c.model == "" {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	opts := llm.NewEmbeddingsOptions(funcs...)

	req := openrouter.EmbeddingsRequest{
		Input: inputs,
		Model: c.model,
	}

	if opts.Dimensions != nil {
		req.Dimensions = opts.Dimensions
	}

	slog.DebugContext(ctx, "starting embeddings")
	before := time.Now()
	res, err := c.client.CreateEmbeddings(ctx, req)
	slog.DebugContext(ctx, "embeddings completed", slog.Duration("duration", time.Since(before)))

	if err != nil {
		var reqErr *openrouter.RequestError
		if errors.As(err, &reqErr) {
			if reqErr.HTTPStatusCode == http.StatusTooManyRequests {
				return nil, errors.WithStack(llm.ErrRateLimit)
			}
		}

		return nil, errors.WithStack(err)
	}

	embeddings := make([][]float64, 0)
	for _, d := range res.Data {
		embeddings = append(embeddings, d.Embedding.Vector)
	}

	var usage llm.EmbeddingsUsage
	if res.Usage != nil {
		usage = llm.NewEmbeddingsUsage(int64(res.Usage.PromptTokens), int64(res.Usage.TotalTokens))
	} else {
		usage = llm.NewEmbeddingsUsage(0, 0)
	}

	return &EmbeddingsResponse{embeddings: embeddings, usage: usage}, nil
}

type EmbeddingsResponse struct {
	embeddings [][]float64
	usage      llm.EmbeddingsUsage
}

// Embeddings implements [llm.EmbeddingsResponse].
func (r *EmbeddingsResponse) Embeddings() [][]float64 {
	return r.embeddings
}

// Usage implements [llm.EmbeddingsResponse].
func (r *EmbeddingsResponse) Usage() llm.EmbeddingsUsage {
	return r.usage
}

func NewEmbeddingsClient(client *openrouter.Client, model string) *EmbeddingsClient {
	return &EmbeddingsClient{
		client: client,
		model:  model,
	}
}

var _ llm.EmbeddingsClient = &EmbeddingsClient{}

var _ llm.EmbeddingsResponse = &EmbeddingsResponse{}

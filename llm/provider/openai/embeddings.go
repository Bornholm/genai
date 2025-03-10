package openai

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"github.com/pkg/errors"
)

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	if c.embeddingsModel == "" {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	opts := llm.NewEmbeddingsOptions(funcs...)

	params := openai.EmbeddingNewParams{
		Input: openai.F[openai.EmbeddingNewParamsInputUnion](shared.UnionString(opts.Input)),
		Model: openai.F(c.embeddingsModel),
	}

	if opts.Dimensions != nil {
		params.Dimensions = openai.Int(int64(*opts.Dimensions))
	}

	res, err := c.client.Embeddings.New(ctx, params)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	embeddings := make([][]float64, 0)
	for _, d := range res.Data {
		embeddings = append(embeddings, d.Embedding)
	}

	return &EmbeddingsResponse{embeddings: embeddings}, nil
}

type EmbeddingsResponse struct {
	embeddings [][]float64
}

// Embeddings implements llm.EmbeddingsResponse.
func (r *EmbeddingsResponse) Embeddings() [][]float64 {
	return r.embeddings
}

var _ llm.EmbeddingsResponse = &EmbeddingsResponse{}

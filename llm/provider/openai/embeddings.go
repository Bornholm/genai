package openai

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"github.com/pkg/errors"
)

type EmbeddingsClient struct {
	client *openai.Client
	model  string
}

// Embeddings implements llm.Client.
func (c *EmbeddingsClient) Embeddings(ctx context.Context, input string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	if c.model == "" {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	opts := llm.NewEmbeddingsOptions(funcs...)

	params := openai.EmbeddingNewParams{
		Input: openai.F[openai.EmbeddingNewParamsInputUnion](shared.UnionString(input)),
		Model: openai.F(c.model),
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

func NewEmbeddingsClient(client *openai.Client, model string) *EmbeddingsClient {
	return &EmbeddingsClient{
		client: client,
		model:  model,
	}
}

var _ llm.EmbeddingsClient = &EmbeddingsClient{}

var _ llm.EmbeddingsResponse = &EmbeddingsResponse{}

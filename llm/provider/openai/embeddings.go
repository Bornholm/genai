package openai

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/pkg/errors"
)

type EmbeddingsClient struct {
	client openai.Client
	model  string
}

// Embeddings implements llm.Client.
func (c *EmbeddingsClient) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	if c.model == "" {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	opts := llm.NewEmbeddingsOptions(funcs...)

	params := openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: inputs,
		},
		Model: openai.EmbeddingModel(c.model),
	}

	if opts.Dimensions != nil {
		params.Dimensions = openai.Int(int64(*opts.Dimensions))
	}

	var httpRes *http.Response

	slog.DebugContext(ctx, "starting embeddings")
	before := time.Now()
	res, err := c.client.Embeddings.New(ctx, params, option.WithResponseInto(&httpRes))
	slog.DebugContext(ctx, "embeddings completed", slog.Duration("duration", time.Since(before)))

	if err != nil {
		if httpRes != nil {
			body, _ := io.ReadAll(httpRes.Body)
			return nil, errors.WithStack(llm.NewHTTPError(httpRes.StatusCode, string(body)))
		}

		return nil, errors.WithStack(err)
	}

	embeddings := make([][]float64, 0)
	for _, d := range res.Data {
		embeddings = append(embeddings, d.Embedding)
	}

	usage := llm.NewEmbeddingsUsage(res.Usage.PromptTokens, res.Usage.TotalTokens)

	return &EmbeddingsResponse{embeddings: embeddings, usage: usage}, nil
}

type EmbeddingsResponse struct {
	embeddings [][]float64
	usage      llm.EmbeddingsUsage
}

// Usage implements llm.EmbeddingsResponse.
func (r *EmbeddingsResponse) Usage() llm.EmbeddingsUsage {
	return r.usage
}

// Embeddings implements llm.EmbeddingsResponse.
func (r *EmbeddingsResponse) Embeddings() [][]float64 {
	return r.embeddings
}

func NewEmbeddingsClient(client openai.Client, model string) *EmbeddingsClient {
	return &EmbeddingsClient{
		client: client,
		model:  model,
	}
}

var _ llm.EmbeddingsClient = &EmbeddingsClient{}

var _ llm.EmbeddingsResponse = &EmbeddingsResponse{}

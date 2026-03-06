package proxy

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"

	"github.com/bornholm/genai/llm"
)

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		writeAPIError(w, NewBadRequestError("could not read request body"))
		return
	}

	model, inputs, embOpts, err := ParseEmbeddingRequest(json.RawMessage(rawBody))
	if err != nil {
		writeAPIError(w, NewBadRequestError(err.Error()))
		return
	}

	req := &ProxyRequest{
		Type:             RequestTypeEmbedding,
		Model:            model,
		Headers:          r.Header,
		Body:             json.RawMessage(rawBody),
		EmbeddingOptions: embOpts,
		Metadata:         make(map[string]any),
	}

	if s.options.AuthExtractor != nil {
		userID, err := s.options.AuthExtractor(r)
		if err != nil {
			writeAPIError(w, NewUnauthorizedError(err.Error()))
			return
		}
		req.UserID = userID
	}

	shortCircuit, err := s.chain.RunPreRequest(ctx, req)
	if err != nil {
		writeAPIError(w, NewInternalError(err.Error()))
		return
	}
	if shortCircuit != nil {
		writeProxyResponse(w, shortCircuit)
		return
	}

	rawClient, resolvedModel, apiErr := s.resolveClient(r, req)
	if apiErr != nil {
		writeAPIError(w, apiErr)
		return
	}

	embClient, ok := rawClient.(llm.EmbeddingsClient)
	if !ok {
		writeAPIError(w, NewInternalError("provider does not implement EmbeddingsClient"))
		return
	}

	llmRes, err := embClient.Embeddings(ctx, inputs, req.EmbeddingOptions...)
	if err != nil {
		slog.ErrorContext(ctx, "embeddings error", slog.Any("error", err))
		errRes, _ := s.chain.RunOnError(ctx, req, err)
		if errRes != nil {
			writeProxyResponse(w, errRes)
		} else {
			writeAPIError(w, apiErrorFromErr(err))
		}
		return
	}

	body := FormatEmbeddingResponse(llmRes, resolvedModel)
	usage := llmRes.Usage()
	proxyRes := &ProxyResponse{
		StatusCode: http.StatusOK,
		Body:       body,
		TokensUsed: &TokenUsage{
			PromptTokens: int(usage.PromptTokens()),
			TotalTokens:  int(usage.TotalTokens()),
		},
	}

	if err := s.chain.RunPostResponse(ctx, req, proxyRes); err != nil {
		slog.WarnContext(ctx, "post-response hook error", slog.Any("error", err))
	}

	writeProxyResponse(w, proxyRes)
}

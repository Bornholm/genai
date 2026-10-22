package proxy

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"

	"github.com/bornholm/genai/llm"
)

func (s *Server) handleDecisions(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		writeAPIError(w, NewBadRequestError("could not read request body"))
		return
	}

	// Parsed here so that a malformed request is refused before any hook
	// runs; what goes upstream is parsed again from the body the hooks leave.
	model, _, _, err := ParseDecisionRequest(json.RawMessage(rawBody))
	if err != nil {
		writeAPIError(w, NewBadRequestError(err.Error()))
		return
	}

	req := &ProxyRequest{
		Type:      RequestTypeDecision,
		Model:     model,
		Headers:   r.Header,
		SessionID: sessionIDFromHeaders(r.Header),
		Body:      json.RawMessage(rawBody),
		Metadata:  make(map[string]any),
	}

	if s.options.AuthExtractor != nil {
		userID, err := s.options.AuthExtractor(r)
		if err != nil {
			writeAPIError(w, authAPIError(r.Context(), err))
			return
		}
		req.UserID = userID
		ctx = r.Context()
	}

	shortCircuit, err := s.chain.RunPreRequest(ctx, req)
	if err != nil {
		writeAPIError(w, hookAPIError(err, "a pre-request hook failed"))
		return
	}
	if shortCircuit != nil {
		writeProxyResponse(w, shortCircuit)
		return
	}

	// The model stays req.Model, the lever of the hooks on every route: the
	// one the body may name now is not read.
	_, state, questions, err := ParseDecisionRequest(req.Body)
	if err != nil {
		slog.ErrorContext(ctx, "a pre-request hook left an invalid decision request", slog.Any("error", err))
		writeAPIError(w, NewInternalError("a pre-request hook failed"))
		return
	}

	rawClient, resolvedModel, errRes, apiErr := s.resolveClient(r, req)
	if errRes != nil {
		writeProxyResponse(w, errRes)
		return
	}
	if apiErr != nil {
		writeAPIError(w, apiErr)
		return
	}

	// The resolver returns an llm.Client, which decision clients are not
	// part of: the capability is checked on what it actually returned. A
	// model that cannot decide is the client's mistake, not an outage; a
	// resolver returning no client is the proxy's.
	if rawClient == nil {
		slog.ErrorContext(ctx, "model resolver returned no client", slog.String("model", req.Model))
		writeAPIError(w, NewInternalError("could not resolve the model"))
		return
	}
	if !llm.SupportsDecision(rawClient) {
		writeAPIError(w, NewBadRequestError("model '"+req.Model+"' does not support decisions"))
		return
	}

	if req.SessionID != "" {
		ctx = llm.ContextWithSessionID(ctx, req.SessionID)
	}

	llmRes, err := llm.Decide(ctx, rawClient, state, questions)
	if err != nil {
		slog.ErrorContext(ctx, "decision error", slog.Any("error", err))
		errRes, _ := s.chain.RunOnError(ctx, req, err)
		if errRes != nil {
			writeProxyResponse(w, errRes)
		} else {
			writeAPIError(w, apiErrorFromErr(err))
		}
		return
	}
	// A resolver may hand back any client, a plugin's included: an empty
	// answer is an upstream failure, not a panic.
	if llmRes == nil {
		slog.ErrorContext(ctx, "decision client returned no response")
		writeAPIError(w, NewInternalError("the provider returned no decision"))
		return
	}

	proxyRes := &ProxyResponse{
		StatusCode: http.StatusOK,
		Body:       FormatDecisionResponse(llmRes, resolvedModel),
		TokensUsed: decisionTokenUsage(llmRes.Usage()),
	}

	if err := s.chain.RunPostResponse(ctx, req, proxyRes); err != nil {
		slog.WarnContext(ctx, "post-response hook error", slog.Any("error", err))
	}

	writeProxyResponse(w, proxyRes)
}

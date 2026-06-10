package proxy

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"

	"github.com/bornholm/genai/llm"
)

// handleMessages implements the Anthropic Messages API (POST /messages).
func (s *Server) handleMessages(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	// 1. Read body
	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicAPIError(w, NewBadRequestError("could not read request body"))
		return
	}

	// 2. Parse
	model, stream, chatOpts, err := ParseMessagesRequest(json.RawMessage(rawBody))
	if err != nil {
		writeAnthropicAPIError(w, NewBadRequestError(err.Error()))
		return
	}

	// 3. Build ProxyRequest
	req := &ProxyRequest{
		Type:        RequestTypeMessage,
		Model:       model,
		Headers:     r.Header,
		Body:        json.RawMessage(rawBody),
		ChatOptions: chatOpts,
		Metadata:    make(map[string]any),
	}

	// 4. Extract UserID
	if s.options.AuthExtractor != nil {
		userID, err := s.options.AuthExtractor(r)
		if err != nil {
			writeAnthropicAPIError(w, NewUnauthorizedError(err.Error()))
			return
		}
		req.UserID = userID
	}

	// 5. Pre-request hooks
	shortCircuit, err := s.chain.RunPreRequest(ctx, req)
	if err != nil {
		writeAnthropicAPIError(w, NewInternalError(err.Error()))
		return
	}
	if shortCircuit != nil {
		writeProxyResponse(w, shortCircuit)
		return
	}

	// 6. Resolve model → client
	rawClient, resolvedModel, apiErr := s.resolveClient(r, req)
	if apiErr != nil {
		writeAnthropicAPIError(w, apiErr)
		return
	}

	// 7. Use potentially updated options from req
	opts := req.ChatOptions

	if stream {
		streamingClient, ok := rawClient.(llm.ChatCompletionStreamingClient)
		if !ok {
			writeAnthropicAPIError(w, NewInternalError("provider does not support streaming"))
			return
		}
		emitter := newAnthropicStreamEmitter(resolvedModel)
		s.streamChatCompletion(w, r, req, streamingClient, resolvedModel, opts, emitter)
		return
	}

	completionClient, ok := rawClient.(llm.ChatCompletionClient)
	if !ok {
		writeAnthropicAPIError(w, NewInternalError("provider does not implement ChatCompletionClient"))
		return
	}

	// 8. Call LLM
	llmRes, err := completionClient.ChatCompletion(ctx, opts...)
	if err != nil {
		slog.ErrorContext(ctx, "messages completion error", slog.Any("error", err))
		errRes, _ := s.chain.RunOnError(ctx, req, err)
		if errRes != nil {
			writeProxyResponse(w, errRes)
		} else {
			writeAnthropicAPIError(w, apiErrorFromErr(err))
		}
		return
	}

	// 9. Build ProxyResponse
	body := FormatMessagesResponse(llmRes, resolvedModel)
	usage := llmRes.Usage()
	tokensUsed := &TokenUsage{
		PromptTokens:     int(usage.PromptTokens()),
		CompletionTokens: int(usage.CompletionTokens()),
		TotalTokens:      int(usage.TotalTokens()),
	}
	type cachedUsage interface{ CachedTokens() int64 }
	if cu, ok := usage.(cachedUsage); ok {
		tokensUsed.CachedTokens = int(cu.CachedTokens())
	}
	proxyRes := &ProxyResponse{
		StatusCode: http.StatusOK,
		Body:       body,
		TokensUsed: tokensUsed,
	}

	// 10. Post-response hooks
	if err := s.chain.RunPostResponse(ctx, req, proxyRes); err != nil {
		slog.WarnContext(ctx, "post-response hook error", slog.Any("error", err))
	}

	// 11. Write response
	writeProxyResponse(w, proxyRes)
}

// handleCountTokens implements the Anthropic token counting API
// (POST /messages/count_tokens). It returns a heuristic estimate and does not
// run the full hook chain (no LLM call, no quota/usage tracking).
func (s *Server) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicAPIError(w, NewBadRequestError("could not read request body"))
		return
	}

	if s.options.AuthExtractor != nil {
		if _, err := s.options.AuthExtractor(r); err != nil {
			writeAnthropicAPIError(w, NewUnauthorizedError(err.Error()))
			return
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"input_tokens": EstimateTokenCount(json.RawMessage(rawBody)),
	})
}

package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"

	"github.com/bornholm/genai/llm"
	"github.com/google/uuid"
)

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	// 1. Read body
	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		writeAPIError(w, NewBadRequestError("could not read request body"))
		return
	}

	// 2. Parse
	model, stream, chatOpts, err := ParseChatCompletionRequest(json.RawMessage(rawBody))
	if err != nil {
		writeAPIError(w, NewBadRequestError(err.Error()))
		return
	}

	// 3. Build ProxyRequest
	req := &ProxyRequest{
		Type:        RequestTypeChatCompletion,
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
			writeAPIError(w, NewUnauthorizedError(err.Error()))
			return
		}
		req.UserID = userID
	}

	// 5. Pre-request hooks
	shortCircuit, err := s.chain.RunPreRequest(ctx, req)
	if err != nil {
		writeAPIError(w, NewInternalError(err.Error()))
		return
	}
	if shortCircuit != nil {
		writeProxyResponse(w, shortCircuit)
		return
	}

	// 6. Resolve model → client
	rawClient, resolvedModel, apiErr := s.resolveClient(r, req)
	if apiErr != nil {
		writeAPIError(w, apiErr)
		return
	}

	// 7. Use potentially updated options from req
	opts := req.ChatOptions

	if stream {
		streamingClient, ok := rawClient.(llm.ChatCompletionStreamingClient)
		if !ok {
			writeAPIError(w, NewInternalError("provider does not support streaming"))
			return
		}
		s.handleChatCompletionsStream(w, r, req, streamingClient, resolvedModel, opts)
		return
	}

	completionClient, ok := rawClient.(llm.ChatCompletionClient)
	if !ok {
		writeAPIError(w, NewInternalError("provider does not implement ChatCompletionClient"))
		return
	}

	// 8. Call LLM
	llmRes, err := completionClient.ChatCompletion(ctx, opts...)
	if err != nil {
		slog.ErrorContext(ctx, "chat completion error", slog.Any("error", err))
		errRes, _ := s.chain.RunOnError(ctx, req, err)
		if errRes != nil {
			writeProxyResponse(w, errRes)
		} else {
			writeAPIError(w, apiErrorFromErr(err))
		}
		return
	}

	// 9. Build ProxyResponse
	body := FormatChatCompletionResponse(llmRes, resolvedModel)
	usage := llmRes.Usage()
	proxyRes := &ProxyResponse{
		StatusCode: http.StatusOK,
		Body:       body,
		TokensUsed: &TokenUsage{
			PromptTokens:     int(usage.PromptTokens()),
			CompletionTokens: int(usage.CompletionTokens()),
			TotalTokens:      int(usage.TotalTokens()),
		},
	}

	// 10. Post-response hooks
	if err := s.chain.RunPostResponse(ctx, req, proxyRes); err != nil {
		slog.WarnContext(ctx, "post-response hook error", slog.Any("error", err))
	}

	// 11. Write response
	writeProxyResponse(w, proxyRes)
}

func (s *Server) handleChatCompletionsStream(
	w http.ResponseWriter,
	r *http.Request,
	req *ProxyRequest,
	client llm.ChatCompletionStreamingClient,
	resolvedModel string,
	opts []llm.ChatCompletionOptionFunc,
) {
	ctx := r.Context()

	chunks, err := client.ChatCompletionStream(ctx, opts...)
	if err != nil {
		slog.ErrorContext(ctx, "stream chat completion error", slog.Any("error", err))
		errRes, _ := s.chain.RunOnError(ctx, req, err)
		if errRes != nil {
			writeProxyResponse(w, errRes)
		} else {
			writeAPIError(w, apiErrorFromErr(err))
		}
		return
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, canFlush := w.(http.Flusher)

	streamID := "chatcmpl-" + uuid.New().String()
	tracker := llm.NewStreamingUsageTracker()

	for chunk := range chunks {
		if chunk.Error() != nil {
			slog.ErrorContext(ctx, "stream chunk error", slog.Any("error", chunk.Error()))
			break
		}

		tracker.Update(chunk)

		payload := FormatStreamChunk(chunk, streamID, resolvedModel)
		data, err := json.Marshal(payload)
		if err != nil {
			slog.ErrorContext(ctx, "could not marshal stream chunk", slog.Any("error", err))
			continue
		}

		fmt.Fprintf(w, "data: %s\n\n", data)
		if canFlush {
			flusher.Flush()
		}

		if chunk.IsComplete() {
			break
		}
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	if canFlush {
		flusher.Flush()
	}

	// Post-response hook with accumulated usage
	usage := tracker.Usage()
	proxyRes := &ProxyResponse{
		StatusCode: http.StatusOK,
		Body:       nil,
		TokensUsed: &TokenUsage{
			PromptTokens:     int(usage.PromptTokens()),
			CompletionTokens: int(usage.CompletionTokens()),
			TotalTokens:      int(usage.TotalTokens()),
		},
	}
	if err := s.chain.RunPostResponse(ctx, req, proxyRes); err != nil {
		slog.WarnContext(ctx, "post-response hook error", slog.Any("error", err))
	}
}

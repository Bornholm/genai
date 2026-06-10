package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"

	"github.com/bornholm/genai/llm"
	"github.com/google/uuid"
	"github.com/pkg/errors"
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

func (s *Server) handleChatCompletionsStream(
	w http.ResponseWriter,
	r *http.Request,
	req *ProxyRequest,
	client llm.ChatCompletionStreamingClient,
	resolvedModel string,
	opts []llm.ChatCompletionOptionFunc,
) {
	emitter := newOpenAIStreamEmitter(resolvedModel)
	s.streamChatCompletion(w, r, req, client, resolvedModel, opts, emitter)
}

// openAIStreamEmitter encodes llm.StreamChunk values as OpenAI-compatible
// "chat.completion.chunk" SSE events.
type openAIStreamEmitter struct {
	streamID     string
	model        string
	sawToolCalls bool
}

func newOpenAIStreamEmitter(model string) *openAIStreamEmitter {
	return &openAIStreamEmitter{
		streamID: "chatcmpl-" + uuid.New().String(),
		model:    model,
	}
}

func (e *openAIStreamEmitter) write(w io.Writer, chunk llm.StreamChunk) error {
	if !chunk.IsComplete() && chunk.Delta() != nil && len(chunk.Delta().ToolCalls()) > 0 {
		e.sawToolCalls = true
	}

	payload := FormatStreamChunk(chunk, e.streamID, e.model, e.sawToolCalls)
	data, err := json.Marshal(payload)
	if err != nil {
		return errors.WithStack(err)
	}

	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return errors.WithStack(err)
	}
	return nil
}

// EmitFirst implements streamEmitter.
func (e *openAIStreamEmitter) EmitFirst(w io.Writer, chunk llm.StreamChunk) error {
	return e.write(w, chunk)
}

// Emit implements streamEmitter.
func (e *openAIStreamEmitter) Emit(w io.Writer, chunk llm.StreamChunk) error {
	return e.write(w, chunk)
}

// EmitError implements streamEmitter.
func (e *openAIStreamEmitter) EmitError(w io.Writer, err error) error {
	apiErr := apiErrorFromErr(err)
	data, merr := json.Marshal(ErrorResponse{Error: *apiErr})
	if merr != nil {
		return errors.WithStack(merr)
	}
	if _, werr := fmt.Fprintf(w, "data: %s\n\n", data); werr != nil {
		return errors.WithStack(werr)
	}
	return nil
}

// Finalize implements streamEmitter.
func (e *openAIStreamEmitter) Finalize(w io.Writer, usage llm.ChatCompletionUsage) error {
	if _, err := fmt.Fprintf(w, "data: [DONE]\n\n"); err != nil {
		return errors.WithStack(err)
	}
	return nil
}

var _ streamEmitter = &openAIStreamEmitter{}

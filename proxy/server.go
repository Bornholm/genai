package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
)

// Server is the OpenAI-compatible proxy HTTP server.
type Server struct {
	options *Options
	chain   *HookChain
	mux     *http.ServeMux
}

// NewServer builds a Server from functional options.
func NewServer(funcs ...OptionFunc) *Server {
	opts := defaultOptions()
	for _, fn := range funcs {
		fn(opts)
	}

	s := &Server{
		options: opts,
		chain:   NewHookChain(opts.Hooks...),
		mux:     http.NewServeMux(),
	}

	s.mux.HandleFunc("POST /chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("POST /embeddings", s.handleEmbeddings)
	s.mux.HandleFunc("GET /models", s.handleModels)
	s.mux.HandleFunc("POST /messages", s.handleMessages)
	s.mux.HandleFunc("POST /messages/count_tokens", s.handleCountTokens)

	return s
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

// ListenAndServe starts the HTTP server.
func (s *Server) ListenAndServe() error {
	slog.Info("proxy server listening", slog.String("addr", s.options.Addr))
	return http.ListenAndServe(s.options.Addr, s)
}

// writeJSON serializes body as JSON and sets the appropriate Content-Type.
// A status outside 200-599, such as the zero StatusCode of an *APIError or a
// ProxyResponse a hook left unset, is answered with a 500: net/http panics on
// a status below 100 and a 1xx is not a final response.
func writeJSON(w http.ResponseWriter, status int, body any) {
	if status < 200 || status > 599 {
		slog.Error("invalid response status, answering with 500", slog.Int("status", status))
		status = http.StatusInternalServerError
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	if err := json.NewEncoder(w).Encode(body); err != nil {
		slog.Error("could not write JSON response", slog.Any("error", err))
	}
}

// writeAPIError writes an OpenAI-compatible error response.
func writeAPIError(w http.ResponseWriter, apiErr *APIError) {
	writeJSON(w, apiErr.StatusCode, ErrorResponse{Error: *apiErr})
}

// writeProxyResponse writes a ProxyResponse to the HTTP response writer.
func writeProxyResponse(w http.ResponseWriter, res *ProxyResponse) {
	if res.Headers != nil {
		for k, vals := range res.Headers {
			for _, v := range vals {
				w.Header().Add(k, v)
			}
		}
	}
	writeJSON(w, res.StatusCode, res.Body)
}

// writeAnthropicProxyResponse writes res like writeProxyResponse, but a body
// holding an OpenAI error envelope, which is how hooks refuse a request, is
// rewritten in the Anthropic one so that /messages clients can read it.
func writeAnthropicProxyResponse(w http.ResponseWriter, res *ProxyResponse) {
	var apiErr *APIError
	switch body := res.Body.(type) {
	case ErrorResponse:
		apiErr = &body.Error
	case *ErrorResponse:
		if body != nil {
			apiErr = &body.Error
		}
	}
	if apiErr == nil {
		writeProxyResponse(w, res)
		return
	}
	converted := *res
	converted.Body = anthropicErrorResponse(apiErr)
	writeProxyResponse(w, &converted)
}

// resolveClient returns the llm.Client to use for req.
// It tries the hook chain resolvers first, then falls back to DefaultClient,
// but only when no resolver handles the model: a resolver error may be a
// refusal, such as a model the user is not allowed to use. Such an error is
// offered to the error hooks, whose response is returned when one answers.
func (s *Server) resolveClient(r *http.Request, req *ProxyRequest) (client interface{}, model string, errRes *ProxyResponse, apiErr *APIError) {
	ctx := r.Context()
	c, m, err := s.chain.ResolveModel(ctx, req)
	if err == nil {
		return c, m, nil, nil
	}

	if !errors.Is(err, ErrNoResolver) {
		slog.ErrorContext(ctx, "model resolver error",
			slog.String("model", req.Model),
			slog.Any("error", err),
		)
		if res, hookErr := s.chain.RunOnError(ctx, req, err); res != nil {
			return nil, "", res, nil
		} else if hookErr != nil {
			slog.ErrorContext(ctx, "error hooks failed", slog.Any("error", hookErr))
		}
		return nil, "", nil, hookAPIError(err, "could not resolve the model")
	}

	if s.options.DefaultClient != nil {
		return s.options.DefaultClient, req.Model, nil, nil
	}

	return nil, "", nil, NewModelNotFoundError(req.Model)
}

// authAPIError answers a request the AuthExtractor refused: with the
// *APIError it failed with, or a generic message otherwise, the cause being
// logged since it may hold internal details, such as a key server address.
func authAPIError(ctx context.Context, err error) *APIError {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr
	}
	slog.WarnContext(ctx, "authentication failed", slog.Any("error", err))
	return NewUnauthorizedError("authentication failed")
}

// hookAPIError returns the *APIError a hook or a resolver failed with, so that
// it can refuse a request with its own status, or a generic internal error
// otherwise: the cause, already logged, may hold internal details. The
// *APIError is looked up through wrapped errors, so a hook may add context to
// it without losing the status.
func hookAPIError(err error, message string) *APIError {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr
	}
	return NewInternalError(message)
}

package proxy

import (
	"encoding/json"
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

	s.mux.HandleFunc("POST /api/v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("POST /api/v1/embeddings", s.handleEmbeddings)
	s.mux.HandleFunc("GET /api/v1/models", s.handleModels)

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
func writeJSON(w http.ResponseWriter, status int, body any) {
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

// resolveClient returns the llm.Client to use for req.
// It tries the hook chain resolvers first, then falls back to DefaultClient.
func (s *Server) resolveClient(r *http.Request, req *ProxyRequest) (client interface{}, model string, apiErr *APIError) {
	c, m, err := s.chain.ResolveModel(r.Context(), req)
	if err == nil {
		return c, m, nil
	}

	if s.options.DefaultClient != nil {
		return s.options.DefaultClient, req.Model, nil
	}

	return nil, "", NewModelNotFoundError(req.Model)
}

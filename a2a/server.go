package a2a

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/bornholm/genai/llm/provider/openrouter"
)

// Server implements the A2A HTTP server
type Server struct {
	card    AgentCard
	handler TaskHandler
	opts    *ServerOptions
	mux     *http.ServeMux
}

// NewServer creates a new A2A server
func NewServer(card AgentCard, handler TaskHandler, funcs ...ServerOptionFunc) *Server {
	opts := NewServerOptions(funcs...)
	s := &Server{
		card:    card,
		handler: handler,
		opts:    opts,
		mux:     http.NewServeMux(),
	}
	s.mux.HandleFunc("GET /.well-known/agent.json", s.handleAgentCard)
	s.mux.HandleFunc("POST /", s.handleJSONRPC)
	return s
}

// ServeHTTP makes Server implement http.Handler so it can be
// composed with middleware or used in existing servers.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *Server) handleAgentCard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if err := json.NewEncoder(w).Encode(s.card); err != nil {
		slog.Error("failed to encode agent card", "error", err)
	}
}

func (s *Server) handleJSONRPC(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers for all responses
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight requests
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	var req JSONRPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONRPCError(w, nil, NewParseError("invalid JSON: "+err.Error()))
		return
	}

	if req.JSONRPC != "2.0" {
		writeJSONRPCError(w, req.ID, NewInvalidRequestError("jsonrpc must be '2.0'"))
		return
	}

	switch req.Method {
	case MethodTasksSend:
		s.handleTasksSend(w, r, req)
	case MethodTasksSendSubscribe:
		s.handleTasksSendSubscribe(w, r, req)
	case MethodTasksGet:
		s.handleTasksGet(w, r, req)
	case MethodTasksCancel:
		s.handleTasksCancel(w, r, req)
	default:
		writeJSONRPCError(w, req.ID, NewMethodNotFoundError(req.Method))
	}
}

func (s *Server) handleTasksSend(w http.ResponseWriter, r *http.Request, req JSONRPCRequest) {
	var params TaskSendParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		writeJSONRPCError(w, req.ID, NewInvalidParamsError(err.Error()))
		return
	}

	ctx := openrouter.WithTransforms(r.Context(), []string{openrouter.TransformMiddleOut})

	task, err := s.handler.HandleTask(ctx, params)
	if err != nil {
		writeJSONRPCError(w, req.ID, NewInternalError(err.Error()))
		return
	}

	writeJSONRPCResult(w, req.ID, task)
}

func (s *Server) handleTasksSendSubscribe(w http.ResponseWriter, r *http.Request, req JSONRPCRequest) {
	var params TaskSendParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		writeJSONRPCError(w, req.ID, NewInvalidParamsError(err.Error()))
		return
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSONRPCError(w, req.ID, NewInternalError("streaming not supported"))
		return
	}

	events := make(chan any, 32)

	go func() {
		ctx := openrouter.WithTransforms(r.Context(), []string{openrouter.TransformMiddleOut})

		if err := s.handler.HandleTaskSubscribe(ctx, params, events); err != nil {
			slog.Error("HandleTaskSubscribe failed", "error", err)
		}
	}()

	for evt := range events {
		data, err := json.Marshal(evt)
		if err != nil {
			slog.Error("Failed to marshal SSE event", "error", err)
			continue
		}

		// Wrap each event in a JSON-RPC response envelope
		rpcResponse := JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  json.RawMessage(data),
		}
		rpcData, _ := json.Marshal(rpcResponse)

		fmt.Fprintf(w, "data: %s\n\n", rpcData)
		flusher.Flush()
	}
}

func (s *Server) handleTasksGet(w http.ResponseWriter, r *http.Request, req JSONRPCRequest) {
	var params TaskQueryParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		writeJSONRPCError(w, req.ID, NewInvalidParamsError(err.Error()))
		return
	}

	task, err := s.handler.GetTask(r.Context(), params)
	if err != nil {
		writeJSONRPCError(w, req.ID, NewTaskNotFoundError(params.ID))
		return
	}

	writeJSONRPCResult(w, req.ID, task)
}

func (s *Server) handleTasksCancel(w http.ResponseWriter, r *http.Request, req JSONRPCRequest) {
	var params TaskQueryParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		writeJSONRPCError(w, req.ID, NewInvalidParamsError(err.Error()))
		return
	}

	task, err := s.handler.CancelTask(r.Context(), params)
	if err != nil {
		writeJSONRPCError(w, req.ID, NewTaskNotFoundError(params.ID))
		return
	}

	writeJSONRPCResult(w, req.ID, task)
}

// Helper functions

func writeJSONRPCResult(w http.ResponseWriter, id any, result any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	})
}

func writeJSONRPCError(w http.ResponseWriter, id any, rpcErr *JSONRPCError) {
	w.Header().Set("Content-Type", "application/json")
	// Map JSON-RPC error codes to HTTP status codes
	httpStatus := http.StatusInternalServerError
	switch rpcErr.Code {
	case ErrCodeParseError, ErrCodeInvalidRequest, ErrCodeInvalidParams:
		httpStatus = http.StatusBadRequest
	case ErrCodeMethodNotFound:
		httpStatus = http.StatusNotFound
	case ErrCodeTaskNotFound:
		httpStatus = http.StatusNotFound
	}
	w.WriteHeader(httpStatus)
	json.NewEncoder(w).Encode(JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   rpcErr,
	})
}

package proxy

import (
	"net/http"

	"github.com/bornholm/genai/llm"
)

// sessionHeaders are the headers coding agents name their conversation in,
// read in this order. The first one set gives ProxyRequest.SessionID.
var sessionHeaders = []string{
	"X-Session-Id",
	"X-Claude-Code-Session-Id", // Claude Code
	"Session_id",               // Codex
	"X-Opencode-Session",       // OpenCode
}

// sessionIDFromHeaders returns the session ID a client sent, or an empty
// string.
func sessionIDFromHeaders(header http.Header) string {
	for _, name := range sessionHeaders {
		if id := header.Get(name); id != "" {
			return id
		}
	}
	return ""
}

// withSession passes req.SessionID upstream: as the SessionID chat
// completion option, and on the context, which a relayed request has no
// options for. It returns r and the options unchanged without a session ID.
func withSession(r *http.Request, req *ProxyRequest) (*http.Request, []llm.ChatCompletionOptionFunc) {
	opts := req.ChatOptions
	if req.SessionID == "" {
		return r, opts
	}
	// Clipped, so that the session option never lands in the backing array
	// of the hooks' slice.
	opts = append(opts[:len(opts):len(opts)], llm.WithSessionID(req.SessionID))
	return r.WithContext(llm.ContextWithSessionID(r.Context(), req.SessionID)), opts
}

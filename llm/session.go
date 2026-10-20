package llm

import "context"

type sessionIDContextKey struct{}

// ContextWithSessionID returns a context carrying the session ID of the
// conversation a request belongs to. The HTTP client of a provider configured
// with a session header sends it there (see provider.CommonOptions).
//
// provider.Client sets it from the SessionID chat completion option; a caller
// that talks to a provider client directly, or relays a request, sets it
// itself. An empty ID leaves ctx unchanged.
func ContextWithSessionID(ctx context.Context, id string) context.Context {
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, sessionIDContextKey{}, id)
}

// SessionIDFromContext returns the session ID ctx carries, or an empty
// string.
func SessionIDFromContext(ctx context.Context) string {
	id, _ := ctx.Value(sessionIDContextKey{}).(string)
	return id
}

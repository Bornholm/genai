package agent

import (
	"slices"
)

// Middleware is a function that wraps a Handler to add functionality
type Middleware func(next Handler) Handler

// chain wraps the handler with the middlewares in reverse order
// so that the first middleware in the list is executed first
func chain(handler Handler, middlewares []Middleware) Handler {
	slices.Reverse(middlewares)
	for _, m := range middlewares {
		handler = m(handler)
	}
	return handler
}

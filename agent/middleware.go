package agent

import (
	"slices"
)

type Middleware func(next Handler) Handler

func chain(handler Handler, middlewares []Middleware) Handler {
	slices.Reverse(middlewares)
	for _, m := range middlewares {
		handler = m(handler)
	}
	return handler
}

package llm

import "errors"

var (
	ErrUnavailable = errors.New("unavailable")
	ErrNoMessage   = errors.New("no message")
)

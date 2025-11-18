package router

import (
	"github.com/bornholm/genai/agent"
	"github.com/pkg/errors"
)

const Default = "__default"

var (
	ErrNoMatch = errors.New("no match")
)

type Handler struct {
	matcher        Matcher
	handlers       map[string]agent.Handler
	defaultHandler string
}

// Handle implements agent.Handler.
func (h *Handler) Handle(input agent.Event, outputs chan agent.Event) error {
	name, err := h.matcher.Match(input)
	if err != nil {
		return errors.WithStack(err)
	}

	if name == "" || name == Default {
		name = h.defaultHandler
	}

	handler, exists := h.handlers[name]
	if !exists {
		return errors.Wrapf(ErrNoMatch, "could not find handler named '%s'", name)
	}

	if err := handler.Handle(input, outputs); err != nil {
		return errors.WithStack(err)
	}

	return nil
}

func (h *Handler) Set(name string, handler agent.Handler) {
	h.handlers[name] = handler
}

func (h *Handler) SetDefault(name string) {
	h.defaultHandler = name
}

var _ agent.Handler = &Handler{}

type Matcher interface {
	Match(input agent.Event) (string, error)
}

type MatchFunc func(input agent.Event) (string, error)

func (fn MatchFunc) Match(input agent.Event) (string, error) {
	return fn(input)
}

func NewHandler(matcher Matcher) *Handler {
	return &Handler{
		matcher:  matcher,
		handlers: make(map[string]agent.Handler),
	}
}

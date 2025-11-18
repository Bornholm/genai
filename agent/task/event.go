package task

import (
	"context"

	"github.com/bornholm/genai/agent"
)

type ThoughtType string

const (
	ThoughtTypeAgent     ThoughtType = "agent"
	ThoughtTypeEvaluator ThoughtType = "evaluator"
)

type ThoughtEvent interface {
	agent.MessageEvent
	Type() ThoughtType
	Thought() string
	Iteration() int
	Origin() agent.MessageEvent
}

type BaseThoughtEvent struct {
	id          agent.EventID
	ctx         context.Context
	thought     string
	thoughtType ThoughtType
	iteration   int
	origin      agent.MessageEvent
}

// ID implements ThoughtEvent.
func (e *BaseThoughtEvent) ID() agent.EventID {
	return e.id
}

// WithContext implements ThoughtEvent.
func (e *BaseThoughtEvent) WithContext(ctx context.Context) agent.Event {
	return &BaseThoughtEvent{
		id:          e.id,
		ctx:         ctx,
		thought:     e.thought,
		thoughtType: e.thoughtType,
		iteration:   e.iteration,
		origin:      e.origin,
	}
}

// Context implements ThoughtEvent.
func (e *BaseThoughtEvent) Context() context.Context {
	return e.ctx
}

// Type implements ThoughtEvent.
func (e *BaseThoughtEvent) Type() ThoughtType {
	return e.thoughtType
}

// Message implements ThoughtEvent.
func (e *BaseThoughtEvent) Message() string {
	return e.thought
}

// Event implements ThoughtEvent.
func (e *BaseThoughtEvent) Event() {}

// Iteration implements ThoughtEvent.
func (e *BaseThoughtEvent) Iteration() int {
	return e.iteration
}

// Origin implements ThoughtEvent.
func (e *BaseThoughtEvent) Origin() agent.MessageEvent {
	return e.origin
}

// Thought implements ThoughtEvent.
func (e *BaseThoughtEvent) Thought() string {
	return e.thought
}

var _ ThoughtEvent = &BaseThoughtEvent{}

func NewThoughtEvent(ctx context.Context, iteration int, thoughtType ThoughtType, thought string, origin agent.MessageEvent) *BaseThoughtEvent {
	return &BaseThoughtEvent{
		id:          agent.NewEventID(),
		ctx:         ctx,
		thoughtType: thoughtType,
		iteration:   iteration,
		thought:     thought,
		origin:      origin,
	}
}

type ResultEvent interface {
	agent.MessageEvent
	Thoughts() []string
	Result() string
	Origin() agent.MessageEvent
}

type BaseResultEvent struct {
	id       agent.EventID
	result   string
	thoughts []string
	origin   agent.MessageEvent
	ctx      context.Context
}

// ID implements ResultEvent.
func (e *BaseResultEvent) ID() agent.EventID {
	return e.id
}

// WithContext implements ResultEvent.
func (e *BaseResultEvent) WithContext(ctx context.Context) agent.Event {
	return &BaseResultEvent{
		id:       e.id,
		ctx:      ctx,
		result:   e.result,
		thoughts: e.thoughts,
		origin:   e.origin,
	}
}

// Context implements ResultEvent.
func (e *BaseResultEvent) Context() context.Context {
	return e.ctx
}

// Message implements ResultEvent.
func (e *BaseResultEvent) Message() string {
	return e.result
}

// Thoughts implements ResultEvent.
func (e *BaseResultEvent) Thoughts() []string {
	return e.thoughts
}

// Event implements ResultEvent.
func (e *BaseResultEvent) Event() {}

// Origin implements ResultEvent.
func (e *BaseResultEvent) Origin() agent.MessageEvent {
	return e.origin
}

// Result implements ResultEvent.
func (e *BaseResultEvent) Result() string {
	return e.result
}

var _ ResultEvent = &BaseResultEvent{}

func NewResultEvent(ctx context.Context, result string, thoughts []string, origin agent.MessageEvent) *BaseResultEvent {
	return &BaseResultEvent{
		id:       agent.NewEventID(),
		ctx:      ctx,
		origin:   origin,
		thoughts: thoughts,
		result:   result,
	}
}

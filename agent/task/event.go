package task

import (
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
	thought     string
	thoughtType ThoughtType
	iteration   int
	origin      agent.MessageEvent
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

func NewThoughtEvent(iteration int, thoughtType ThoughtType, thought string, origin agent.MessageEvent) *BaseThoughtEvent {
	return &BaseThoughtEvent{
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
	result   string
	thoughts []string
	origin   agent.MessageEvent
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

func NewResultEvent(result string, thoughts []string, origin agent.MessageEvent) *BaseResultEvent {
	return &BaseResultEvent{
		origin:   origin,
		thoughts: thoughts,
		result:   result,
	}
}

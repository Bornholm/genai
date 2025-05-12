package task

import (
	"github.com/bornholm/genai/agent"
)

type ThoughtEvent interface {
	agent.Event
	Thought() string
	Index() int
	Origin() agent.MessageEvent
}

type BaseThoughtEvent struct {
	thought string
	index   int
	origin  agent.MessageEvent
}

// Event implements ThoughtEvent.
func (e *BaseThoughtEvent) Event() {}

// Index implements ThoughtEvent.
func (e *BaseThoughtEvent) Index() int {
	return e.index
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

func NewThoughtEvent(index int, thought string, origin agent.MessageEvent) *BaseThoughtEvent {
	return &BaseThoughtEvent{
		index:   index,
		thought: thought,
		origin:  origin,
	}
}

type ResultEvent interface {
	agent.Event
	Thoughts() []string
	Result() string
	Origin() agent.MessageEvent
}

type BaseResultEvent struct {
	result   string
	thoughts []string
	origin   agent.MessageEvent
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

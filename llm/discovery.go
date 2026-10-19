package llm

import (
	"context"
	"time"
)

// DiscoveryClient lists the models a provider serves.
//
// Like [ImageGenerationClient], it is deliberately not a member of [Client]:
// callers discover the capability with a type assertion on the client
// returned by provider.Create.
//
//	if discovery, ok := client.(llm.DiscoveryClient); ok {
//	    models, err := discovery.Models(ctx)
//	    // ...
//	}
//
// A discovery client needs no model: the configured one, if any, is ignored.
type DiscoveryClient interface {
	Models(ctx context.Context) ([]ModelInfo, error)
}

// ModelInfo describes a model as its provider lists it. Only the ID is
// always known; the other fields are zero when the provider does not say.
type ModelInfo interface {
	// ID is the name to send as the model of a request.
	ID() string
	// DisplayName is a human-readable name, or empty.
	DisplayName() string
	// CreatedAt is when the provider published the model, or the zero time.
	CreatedAt() time.Time
	// ContextWindow is the maximum number of input tokens, or zero.
	ContextWindow() int64
	// MaxOutputTokens is the maximum number of generated tokens, or zero.
	MaxOutputTokens() int64
}

// BaseModelInfo is the plain implementation of [ModelInfo].
type BaseModelInfo struct {
	id              string
	displayName     string
	createdAt       time.Time
	contextWindow   int64
	maxOutputTokens int64
}

// ID implements [ModelInfo].
func (m *BaseModelInfo) ID() string {
	return m.id
}

// DisplayName implements [ModelInfo].
func (m *BaseModelInfo) DisplayName() string {
	return m.displayName
}

// CreatedAt implements [ModelInfo].
func (m *BaseModelInfo) CreatedAt() time.Time {
	return m.createdAt
}

// ContextWindow implements [ModelInfo].
func (m *BaseModelInfo) ContextWindow() int64 {
	return m.contextWindow
}

// MaxOutputTokens implements [ModelInfo].
func (m *BaseModelInfo) MaxOutputTokens() int64 {
	return m.maxOutputTokens
}

type ModelInfoOptionFunc func(m *BaseModelInfo)

func WithModelDisplayName(name string) ModelInfoOptionFunc {
	return func(m *BaseModelInfo) {
		m.displayName = name
	}
}

func WithModelCreatedAt(createdAt time.Time) ModelInfoOptionFunc {
	return func(m *BaseModelInfo) {
		m.createdAt = createdAt
	}
}

func WithModelContextWindow(tokens int64) ModelInfoOptionFunc {
	return func(m *BaseModelInfo) {
		m.contextWindow = tokens
	}
}

func WithModelMaxOutputTokens(tokens int64) ModelInfoOptionFunc {
	return func(m *BaseModelInfo) {
		m.maxOutputTokens = tokens
	}
}

func NewModelInfo(id string, funcs ...ModelInfoOptionFunc) *BaseModelInfo {
	m := &BaseModelInfo{id: id}
	for _, fn := range funcs {
		fn(m)
	}
	return m
}

var _ ModelInfo = &BaseModelInfo{}

package conformance

import (
	"testing"
)

// Feature flags declare which capabilities the provider under test supports.
type Feature uint64

const (
	FeatureChatCompletion Feature = 1 << iota
	FeatureStreaming
	FeatureToolCalls
	FeatureJSON
	FeatureMultimodal
	FeatureReasoning
	FeatureEmbeddings
)

// Suite runs a set of conformance tests against an llm.Client.
type Suite struct {
	client   any
	features Feature
}

// Option configures a Suite.
type Option func(*Suite)

// WithFeatures sets the feature flags for the suite.
func WithFeatures(f Feature) Option {
	return func(s *Suite) {
		s.features = f
	}
}

// New creates a Suite for the given client.
func New(client any, opts ...Option) *Suite {
	s := &Suite{client: client}
	for _, o := range opts {
		o(s)
	}
	return s
}

func (s *Suite) has(f Feature) bool {
	return s.features&f != 0
}

// Run executes all enabled feature tests as subtests of t.
func (s *Suite) Run(t *testing.T) {
	t.Helper()

	if s.has(FeatureChatCompletion) {
		t.Run("ChatCompletion", func(t *testing.T) {
			testChatCompletion(t, s.client)
		})
	}
	if s.has(FeatureStreaming) {
		t.Run("Streaming", func(t *testing.T) {
			testStreaming(t, s.client)
		})
	}
	if s.has(FeatureToolCalls) {
		t.Run("ToolCalls", func(t *testing.T) {
			testToolCalls(t, s.client)
		})
	}
	if s.has(FeatureJSON) {
		t.Run("JSON", func(t *testing.T) {
			testJSONResponse(t, s.client)
		})
	}
	if s.has(FeatureMultimodal) {
		t.Run("Multimodal", func(t *testing.T) {
			testMultimodal(t, s.client)
		})
	}
	if s.has(FeatureReasoning) {
		t.Run("Reasoning", func(t *testing.T) {
			testReasoning(t, s.client)
		})
	}
	if s.has(FeatureEmbeddings) {
		t.Run("Embeddings", func(t *testing.T) {
			testEmbeddings(t, s.client)
		})
	}
}

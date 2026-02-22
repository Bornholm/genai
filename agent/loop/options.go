package loop

import (
	"github.com/bornholm/genai/llm"
)

const (
	DefaultMaxIterations = 100
	DefaultMaxTokens     = 120000
)

// TruncationStrategy is a function that takes messages and returns a truncated list or an error
type TruncationStrategy func(messages []llm.Message) ([]llm.Message, error)

// Options contains configuration for the loop handler
type Options struct {
	Client              llm.ChatCompletionClient
	Tools               []llm.Tool
	SystemPrompt        string
	MaxIterations       int
	MaxTokens           int
	TokenEstimator      func(string) int
	TruncationStrategy  TruncationStrategy
	ApprovalFunc        ApprovalFunc
	ApprovalRequired    map[string]bool
	ApprovalRequiredAll bool
}

// OptionFunc is a function that configures the loop handler
type OptionFunc func(*Options)

// WithClient sets the LLM client
func WithClient(client llm.ChatCompletionClient) OptionFunc {
	return func(o *Options) {
		o.Client = client
	}
}

// WithTools sets the tools available to the LLM
func WithTools(tools ...llm.Tool) OptionFunc {
	return func(o *Options) {
		o.Tools = tools
	}
}

// WithSystemPrompt sets the system prompt
func WithSystemPrompt(prompt string) OptionFunc {
	return func(o *Options) {
		o.SystemPrompt = prompt
	}
}

// WithMaxIterations sets the maximum number of iterations
func WithMaxIterations(max int) OptionFunc {
	return func(o *Options) {
		o.MaxIterations = max
	}
}

// WithMaxTokens sets the maximum number of tokens for context window management
func WithMaxTokens(max int) OptionFunc {
	return func(o *Options) {
		o.MaxTokens = max
	}
}

// WithTokenEstimator sets the token estimator function
func WithTokenEstimator(estimator func(string) int) OptionFunc {
	return func(o *Options) {
		o.TokenEstimator = estimator
	}
}

// WithTruncationStrategy sets the truncation strategy function
func WithTruncationStrategy(strategy TruncationStrategy) OptionFunc {
	return func(o *Options) {
		o.TruncationStrategy = strategy
	}
}

// ApprovalFunc is called to approve tool execution
type ApprovalFunc func(ctx interface{ Done() <-chan struct{} }, toolName string, arguments string) (bool, error)

// WithApprovalFunc sets the approval function
func WithApprovalFunc(fn ApprovalFunc) OptionFunc {
	return func(o *Options) {
		o.ApprovalFunc = fn
	}
}

// WithApprovalRequiredTools specifies which tools require approval
func WithApprovalRequiredTools(tools ...string) OptionFunc {
	return func(o *Options) {
		if o.ApprovalRequired == nil {
			o.ApprovalRequired = make(map[string]bool)
		}
		for _, t := range tools {
			if t == "*" {
				o.ApprovalRequiredAll = true
			} else {
				o.ApprovalRequired[t] = true
			}
		}
	}
}

// NewOptions creates a new Options with defaults
func NewOptions(funcs ...OptionFunc) *Options {
	opts := &Options{
		MaxIterations:    DefaultMaxIterations,
		MaxTokens:        DefaultMaxTokens,
		Tools:            []llm.Tool{},
		TokenEstimator:   defaultTokenEstimator,
		ApprovalRequired: make(map[string]bool),
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

// defaultTokenEstimator provides a simple heuristic for token estimation
func defaultTokenEstimator(s string) int {
	return len(s) / 4
}

package loop

import (
	"github.com/bornholm/genai/llm"
)

// WithReasoningOptions enables reasoning tokens for each LLM call in the loop.
// When set, reasoning is preserved across tool-call turns automatically.
func WithReasoningOptions(opts *llm.ReasoningOptions) OptionFunc {
	return func(o *Options) {
		o.Reasoning = opts
	}
}

const (
	DefaultMaxIterations = 100
	// DefaultMaxTokens is set to 80000 to leave a safety margin for:
	// - Models with smaller context windows (e.g., Mistral's 131072)
	// - Token estimation errors (rough heuristic)
	// - Budget message overhead (~50 tokens per iteration)
	// - Response generation tokens
	DefaultMaxTokens = 80000
)

// TruncationStrategy is a function that takes messages and returns a truncated list or an error
type TruncationStrategy func(messages []llm.Message) ([]llm.Message, error)

// Options contains configuration for the loop handler
type Options struct {
	Client        llm.ChatCompletionClient
	Tools         []llm.Tool
	SystemPrompt  string
	MaxIterations int
	MaxTokens     int
	// MaxToolResultTokens limits the size of tool results to prevent context overflow.
	// If 0, defaults to 10000 tokens (~40000 chars).
	MaxToolResultTokens int
	TokenEstimator      func(string) int
	TruncationStrategy  TruncationStrategy
	ApprovalFunc        ApprovalFunc
	ApprovalRequired    map[string]bool
	ApprovalRequiredAll bool
	Reasoning           *llm.ReasoningOptions
	// ForcePlanningStep controls whether the handler makes a dedicated first API
	// call with only the TodoWrite tool exposed (and tool_choice=required).
	// This guarantees the model writes a structured plan before taking any action.
	// Defaults to true when tools are present. Set to false to disable.
	ForcePlanningStep bool
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

// WithMaxToolResultTokens sets the maximum number of tokens for tool results
func WithMaxToolResultTokens(max int) OptionFunc {
	return func(o *Options) {
		o.MaxToolResultTokens = max
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

// WithForcePlanningStep enables or disables the dedicated forced TodoWrite planning
// step at the start of each agent task. When enabled (the default), the handler
// makes one extra LLM call before the main loop that exposes only the TodoWrite
// tool with tool_choice=required, guaranteeing the model writes a plan first.
func WithForcePlanningStep(enabled bool) OptionFunc {
	return func(o *Options) {
		o.ForcePlanningStep = enabled
	}
}

// NewOptions creates a new Options with defaults
func NewOptions(funcs ...OptionFunc) *Options {
	opts := &Options{
		MaxIterations:       DefaultMaxIterations,
		MaxTokens:           DefaultMaxTokens,
		MaxToolResultTokens: 10000, // Default to 10K tokens per tool result
		Tools:               []llm.Tool{},
		TokenEstimator:      defaultTokenEstimator,
		ApprovalRequired:    make(map[string]bool),
		ForcePlanningStep:   false,
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

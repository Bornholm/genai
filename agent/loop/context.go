package loop

import (
	"github.com/bornholm/genai/llm"
)

// ContextManager manages the message history to prevent exceeding the context window
type ContextManager struct {
	maxTokens          int
	tokenEstimator     func(string) int
	truncationStrategy TruncationStrategy
}

// NewContextManager creates a new ContextManager
func NewContextManager(maxTokens int, tokenEstimator func(string) int, truncationStrategy TruncationStrategy) *ContextManager {
	if truncationStrategy == nil {
		truncationStrategy = DefaultMiddleOutTruncationStrategy(maxTokens, tokenEstimator)
	}
	return &ContextManager{
		maxTokens:          maxTokens,
		tokenEstimator:     tokenEstimator,
		truncationStrategy: truncationStrategy,
	}
}

// Manage applies the truncation strategy to keep messages under the token limit
func (cm *ContextManager) Manage(messages []llm.Message) []llm.Message {
	if cm.truncationStrategy == nil {
		return messages
	}
	result, err := cm.truncationStrategy(messages)
	if err != nil {
		// On error, return messages unchanged
		return messages
	}
	return result
}

// messageGroup is an atomic unit of messages that must not be split during truncation.
// A ToolCallsMessage and all its following RoleTool messages form one group;
// any other message is a group of one.
type messageGroup []llm.Message

func (g messageGroup) tokens(estimator func(string) int) int {
	total := 0
	for _, m := range g {
		total += estimateMessageTokens(m, estimator)
	}
	return total
}

// groupMessages partitions messages into atomic groups so that a tool_calls
// message is never separated from its tool result messages.
func groupMessages(messages []llm.Message) []messageGroup {
	groups := make([]messageGroup, 0, len(messages))
	i := 0
	for i < len(messages) {
		msg := messages[i]
		if _, ok := msg.(llm.ToolCallsMessage); ok {
			// Gather the tool_calls message plus all immediately following tool results.
			group := messageGroup{msg}
			i++
			for i < len(messages) && messages[i].Role() == llm.RoleTool {
				group = append(group, messages[i])
				i++
			}
			groups = append(groups, group)
		} else {
			groups = append(groups, messageGroup{msg})
			i++
		}
	}
	return groups
}

// DefaultMiddleOutTruncationStrategy creates a middle-out truncation strategy
// that keeps the system message, first user message, and most recent messages.
// It removes complete atomic groups (tool_calls + tool_results) from the middle
// so that API invariants are never violated.
func DefaultMiddleOutTruncationStrategy(maxTokens int, tokenEstimator func(string) int) TruncationStrategy {
	return func(messages []llm.Message) ([]llm.Message, error) {
		if len(messages) <= 2 {
			return messages, nil
		}

		// Estimate total tokens
		totalTokens := estimateMessagesTokens(messages, tokenEstimator)

		// If under the limit, return unchanged
		if totalTokens <= maxTokens {
			return messages, nil
		}

		// Always keep messages[0] (system) and messages[1] (first user) as anchors.
		// The rest is divided into atomic groups; we remove groups from the front of
		// the middle section until we are under budget.
		anchors := messages[:2]
		middle := messages[2:]

		groups := groupMessages(middle)

		// Drop groups from the front until we are within budget.
		excessTokens := totalTokens - maxTokens
		firstKept := 0
		for firstKept < len(groups) && excessTokens > 0 {
			removed := groups[firstKept].tokens(tokenEstimator)
			excessTokens -= removed
			firstKept++
		}

		// If everything in the middle was dropped, return only the anchors.
		if firstKept >= len(groups) {
			return anchors, nil
		}

		// Flatten remaining groups back into a message slice.
		remaining := make([]llm.Message, 0, len(middle))
		for _, g := range groups[firstKept:] {
			remaining = append(remaining, g...)
		}

		result := make([]llm.Message, 0, 2+1+len(remaining))
		result = append(result, anchors...)
		result = append(result, llm.NewMessage(llm.RoleSystem,
			"[Earlier conversation truncated for context limits. The conversation continued from here.]"))
		result = append(result, remaining...)

		return result, nil
	}
}

// NoTruncationStrategy returns a strategy that never truncates
func NoTruncationStrategy() TruncationStrategy {
	return func(messages []llm.Message) ([]llm.Message, error) {
		return messages, nil
	}
}

// estimateMessagesTokens estimates the total tokens in a list of messages
func estimateMessagesTokens(messages []llm.Message, tokenEstimator func(string) int) int {
	total := 0
	for _, msg := range messages {
		total += estimateMessageTokens(msg, tokenEstimator)
	}
	return total
}

// estimateMessageTokens estimates the tokens in a single message
func estimateMessageTokens(msg llm.Message, tokenEstimator func(string) int) int {
	tokens := tokenEstimator(msg.Content())

	// Add overhead for role and formatting
	tokens += 4 // Approximate overhead per message

	// Add tokens for attachments (rough estimate)
	for _, att := range msg.Attachments() {
		// Attachments typically use more tokens
		tokens += tokenEstimator(att.Data()) / 2
	}

	return tokens
}

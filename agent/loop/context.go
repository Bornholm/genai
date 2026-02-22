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

// DefaultMiddleOutTruncationStrategy creates a middle-out truncation strategy
// that keeps the system message, first user message, and most recent messages
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

		// Apply middle-out truncation
		// Keep: system message, first user message, last N messages
		// Replace middle with truncation notice

		// Always keep the first message (system) and second message (first user)
		if len(messages) < 3 {
			return messages, nil
		}

		// Calculate how many tokens we need to remove
		excessTokens := totalTokens - maxTokens

		// Start removing from the middle
		// We keep messages[0] (system), messages[1] (first user), and work backwards from the end

		// Find the middle point
		middleStart := 2
		middleEnd := len(messages) - 1

		// Keep removing from the middle until we're under budget
		for middleStart < middleEnd && excessTokens > 0 {
			// Remove from the start of the middle section
			tokens := estimateMessageTokens(messages[middleStart], tokenEstimator)
			excessTokens -= tokens
			middleStart++
		}

		// If we've removed everything in the middle, just return what we have
		if middleStart >= middleEnd {
			return messages[:2], nil
		}

		// Build the truncated message list
		result := make([]llm.Message, 0, middleEnd-middleStart+3)

		// Add system message
		result = append(result, messages[0])

		// Add first user message
		result = append(result, messages[1])

		// Add truncation notice
		result = append(result, llm.NewMessage(llm.RoleSystem,
			"[Earlier conversation truncated for context limits. The conversation continued from here.]"))

		// Add remaining messages from the end
		result = append(result, messages[middleStart:]...)

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

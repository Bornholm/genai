package messageutil

import (
	"slices"

	"github.com/bornholm/genai/llm"
)

func WithRoles(messages []llm.Message, roles ...llm.Role) []llm.Message {
	return slices.Collect(func(yield func(llm.Message) bool) {
		for _, m := range messages {
			if !slices.Contains(roles, m.Role()) {
				continue
			}
			if !yield(m) {
				return
			}
		}
	})
}

func WithoutRoles(messages []llm.Message, roles ...llm.Role) []llm.Message {
	return slices.Collect(func(yield func(llm.Message) bool) {
		for _, m := range messages {
			if slices.Contains(roles, m.Role()) {
				continue
			}
			if !yield(m) {
				return
			}
		}
	})
}

func InjectSystemPrompt(messages []llm.Message, systemPrompt string) []llm.Message {
	var systemMessage llm.Message = llm.NewMessage(llm.RoleSystem, systemPrompt)

	if len(messages) == 0 {
		messages = append(messages, systemMessage)
		return messages
	}

	firstAssistantMessageIndex := len(messages) - 1
	firstUserMessageIndex := len(messages) - 1
	alreadyExists := false
	for i, m := range messages {
		if m.Role() == llm.RoleAssistant && firstAssistantMessageIndex > i {
			firstAssistantMessageIndex = i
		}

		if m.Role() == llm.RoleUser && firstUserMessageIndex > i {
			firstUserMessageIndex = i
		}

		if m.Role() == llm.RoleSystem && m.Content() == systemPrompt {
			alreadyExists = true
			break
		}
	}

	if alreadyExists {
		return messages
	}

	messages = slices.Insert(messages, min(firstAssistantMessageIndex, firstUserMessageIndex), systemMessage)

	return messages
}

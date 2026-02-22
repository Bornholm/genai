package agent

import (
	"github.com/bornholm/genai/llm"
)

// Input represents a user turn in the conversation.
// A user turn is a text message with zero or more file attachments.
type Input struct {
	Message     string
	Attachments []llm.Attachment
}

// NewInput creates a new Input with the given message and optional attachments
func NewInput(message string, attachments ...llm.Attachment) Input {
	return Input{
		Message:     message,
		Attachments: attachments,
	}
}

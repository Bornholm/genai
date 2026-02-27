package a2a

import (
	"context"
	"fmt"

	"github.com/bornholm/genai/llm"
)

// NewRemoteAgentTool creates an LLM tool that delegates to a remote A2A agent.
// This allows composing agents: one agent can call another as a tool.
func NewRemoteAgentTool(name, description, baseURL string) llm.Tool {
	return &remoteAgentTool{
		name:        name,
		description: description,
		client:      NewClient(baseURL),
	}
}

type remoteAgentTool struct {
	name        string
	description string
	client      *Client
}

// Name implements llm.Tool
func (t *remoteAgentTool) Name() string {
	return t.name
}

// Description implements llm.Tool
func (t *remoteAgentTool) Description() string {
	return t.description
}

// Parameters implements llm.Tool
func (t *remoteAgentTool) Parameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"message": map[string]any{
				"type":        "string",
				"description": "The message to send to the remote agent",
			},
		},
		"required": []string{"message"},
	}
}

// Execute implements llm.Tool
func (t *remoteAgentTool) Execute(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
	message, _ := params["message"].(string)
	if message == "" {
		return nil, fmt.Errorf("message parameter is required")
	}

	task, err := t.client.SendTask(ctx, TaskSendParams{
		Message: Message{
			Role:  "user",
			Parts: []Part{{Type: "text", Text: message}},
		},
	})
	if err != nil {
		return nil, err
	}

	// Extract text from artifacts
	var result string
	for _, artifact := range task.Artifacts {
		for _, part := range artifact.Parts {
			if part.Type == "text" {
				result += part.Text + "\n"
			}
		}
	}

	if result == "" {
		result = fmt.Sprintf("Task completed with status: %s", task.Status.State)
	}

	return llm.NewToolResult(result), nil
}

var _ llm.Tool = &remoteAgentTool{}

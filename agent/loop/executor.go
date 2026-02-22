package loop

import (
	"context"
	"fmt"

	"github.com/bornholm/genai/llm"
)

// executeTool executes a tool call and returns the result as a string
func executeTool(ctx context.Context, tools []llm.Tool, call llm.ToolCall) (string, error) {
	var tool llm.Tool
	for _, t := range tools {
		if call.Name() == t.Name() {
			tool = t
			break
		}
	}

	if tool == nil {
		// Return an error message that the LLM can read and correct from
		availableTools := make([]string, len(tools))
		for i, t := range tools {
			availableTools[i] = t.Name()
		}
		return fmt.Sprintf("Unknown tool: %s. Available tools: %v", call.Name(), availableTools), nil
	}

	result, err := llm.ExecuteToolCall(ctx, call, tool)
	if err != nil {
		// Convert error to a result string - the loop continues
		return fmt.Sprintf("Tool execution error: %s", err.Error()), nil
	}

	return result.Content(), nil
}

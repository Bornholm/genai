package loop

import (
	"embed"
	"io/fs"

	"github.com/bornholm/genai/llm/prompt"
	"github.com/pkg/errors"
)

//go:embed prompts/*.gotmpl
var prompts embed.FS

// RenderSystemPrompt renders the system prompt template with the given data
func RenderSystemPrompt(tmpl string, data map[string]any) (string, error) {
	return prompt.Template(tmpl, data)
}

// RenderSystemPromptFromFS renders a system prompt from the embedded filesystem
func RenderSystemPromptFromFS(filename string, data map[string]any) (string, error) {
	return prompt.FromFS(&prompts, "prompts/"+filename, data)
}

// builtinToolInfos returns the ToolInfo entries for tools that are always
// injected by the loop handler (TodoWrite, TodoRead). These must appear in
// the system prompt's tool listing even though the caller never adds them to
// the tool slice it passes to NewHandler.
func builtinToolInfos() []ToolInfo {
	return []ToolInfo{
		{
			Name:        "TodoWrite",
			Description: "Replace the entire todo list. Use this to create and update your task list. Provide a full JSON array of items (id, content, status). This is a full replacement, not a patch.",
		},
		{
			Name:        "TodoRead",
			Description: "Returns the current todo list as JSON. Use this to check your progress and remaining tasks.",
		},
	}
}

// DefaultSystemPrompt returns the default system prompt.
// The built-in TodoWrite and TodoRead tools are automatically prepended to the
// tool list so they appear in the "Available Tools" section even when the caller
// only passes MCP / user-defined tools.
func DefaultSystemPrompt(tools []ToolInfo, additionalContext string) (string, error) {
	// Prepend built-in tools so the model always sees them in the listing.
	allTools := append(builtinToolInfos(), tools...)
	return RenderSystemPromptFromFS("system.gotmpl", map[string]any{
		"Tools":             allTools,
		"AdditionalContext": additionalContext,
	})
}

// ToolInfo represents tool information for the system prompt
type ToolInfo struct {
	Name        string
	Description string
}

// GetDefaultSystemPrompt returns the raw default system prompt template
func GetDefaultSystemPrompt() (string, error) {
	file, err := prompts.Open("prompts/system.gotmpl")
	if err != nil {
		return "", errors.WithStack(err)
	}
	defer file.Close()

	raw, err := fs.ReadFile(prompts, "prompts/system.gotmpl")
	if err != nil {
		return "", errors.WithStack(err)
	}

	return string(raw), nil
}

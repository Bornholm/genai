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

// DefaultSystemPrompt returns the default system prompt
func DefaultSystemPrompt(tools []ToolInfo, additionalContext string) (string, error) {
	return RenderSystemPromptFromFS("system.gotmpl", map[string]any{
		"Tools":             tools,
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

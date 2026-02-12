package common

import (
	"encoding/json"
	"os"
	"strings"

	llmPrompt "github.com/bornholm/genai/llm/prompt"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"
)

func GetPrompt(ctx *cli.Context, promptParam, dataParam string) (string, error) {
	prompt := ctx.String(promptParam)
	data := ctx.String(dataParam)

	// Check if promptText starts with "@" (file path)
	if strings.HasPrefix(prompt, "@") {
		filePath := prompt[1:] // Remove the "@" prefix
		content, err := os.ReadFile(filePath)
		if err != nil {
			return "", errors.Wrapf(err, "failed to read data file: %s", filePath)
		}
		prompt = string(content)
	}

	if data == "" {
		return prompt, nil
	}

	// Check if dataInput starts with "@" (file path)
	if strings.HasPrefix(data, "@") {
		filePath := data[1:] // Remove the "@" prefix
		content, err := os.ReadFile(filePath)
		if err != nil {
			return "", errors.Wrapf(err, "failed to read data file: %s", filePath)
		}
		data = string(content)
	}

	var templateData any
	if err := json.Unmarshal([]byte(data), &templateData); err != nil {
		return "", errors.Wrap(err, "failed to parse data JSON")
	}

	processed, err := llmPrompt.Template(prompt, templateData)
	if err != nil {
		return "", errors.Wrap(err, "failed to process prompt template")
	}

	return processed, nil
}

package common

import (
	"encoding/json"
	"os"
	"strings"

	llmPrompt "github.com/bornholm/genai/llm/prompt"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"
)

func GetPromptFromContext(ctx *cli.Context, promptParam, dataParam string) (string, error) {
	prompt := ctx.String(promptParam)
	data := ctx.String(dataParam)
	return GetPrompt(ctx, prompt, data)
}

func GetPrompt(ctx *cli.Context, prompt string, rawData string) (string, error) {
	// Check if promptText starts with "@" (file path)
	if strings.HasPrefix(prompt, "@") {
		filePath := prompt[1:] // Remove the "@" prefix
		content, err := os.ReadFile(filePath)
		if err != nil {
			return "", errors.Wrapf(err, "failed to read data file: %s", filePath)
		}
		prompt = string(content)
	}

	if rawData == "" {
		return prompt, nil
	}

	// Check if dataInput starts with "@" (file path)
	if strings.HasPrefix(rawData, "@") {
		filePath := rawData[1:] // Remove the "@" prefix
		content, err := os.ReadFile(filePath)
		if err != nil {
			return "", errors.Wrapf(err, "failed to read data file: %s", filePath)
		}
		rawData = string(content)
	}

	if rawData == "" {
		return prompt, nil
	}

	var templateData any
	if err := json.Unmarshal([]byte(rawData), &templateData); err != nil {
		return "", errors.Wrap(err, "failed to parse data JSON")
	}

	processed, err := llmPrompt.Template(prompt, templateData)
	if err != nil {
		return "", errors.Wrap(err, "failed to process prompt template")
	}

	return processed, nil
}

func GetPromptWithData(ctx *cli.Context, prompt string, data any) (string, error) {
	if prompt == "" {
		return "", nil
	}

	// Check if promptText starts with "@" (file path)
	if strings.HasPrefix(prompt, "@") {
		filePath := prompt[1:]
		content, err := os.ReadFile(filePath)
		if err != nil {
			return "", errors.Wrapf(err, "failed to read prompt file: %s", filePath)
		}
		prompt = string(content)
	}

	if data == nil {
		return prompt, nil
	}

	processed, err := llmPrompt.Template(prompt, data)
	if err != nil {
		return "", errors.Wrap(err, "failed to process prompt template")
	}

	return processed, nil
}

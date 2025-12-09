package llm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"os"
	"path/filepath"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/prompt"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/openrouter"
	"github.com/invopop/jsonschema"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"

	// Import all provider implementations
	_ "github.com/bornholm/genai/llm/provider/all"
)

func Generate() *cli.Command {
	return &cli.Command{
		Name:    "generate",
		Aliases: []string{"gen"},
		Usage:   "Generate chat completion",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:     "system",
				Usage:    "System prompt (text format, or @file to load from file)",
				EnvVars:  []string{"GENAI_SYSTEM_PROMPT"},
				Required: false,
			},
			&cli.StringFlag{
				Name:    "system-data",
				Usage:   "Data to inject in the system prompt (JSON format, or @file to load from file)",
				EnvVars: []string{"GENAI_SYSTEM_DATA"},
			},
			&cli.StringFlag{
				Name:     "prompt",
				Usage:    "User prompt (text format, or @file to load from file)",
				EnvVars:  []string{"GENAI_USER_PROMPT"},
				Required: true,
			},
			&cli.StringFlag{
				Name:    "prompt-data",
				Usage:   "Data to inject in the user prompt (JSON format, or @file to load from file)",
				EnvVars: []string{"GENAI_PROMPT_DATA"},
			},
			&cli.StringSliceFlag{
				Name:    "file",
				Usage:   "File attachments (can be specified multiple times)",
				EnvVars: []string{"GENAI_FILES"},
			},
			&cli.StringFlag{
				Name:    "schema",
				Usage:   "JSON schema file path for structured response",
				EnvVars: []string{"GENAI_SCHEMA"},
			},
			&cli.Float64Flag{
				Name:    "temperature",
				Usage:   "Temperature for generation (0.0 to 2.0)",
				EnvVars: []string{"GENAI_TEMPERATURE"},
				Value:   0.7,
			},
			&cli.StringFlag{
				Name:    "env-file",
				Usage:   "Environment file path",
				EnvVars: []string{"GENAI_ENV_FILE"},
				Value:   ".env",
			},
			&cli.StringFlag{
				Name:    "output",
				Aliases: []string{"o"},
				Usage:   "Output file path (default: stdout)",
				EnvVars: []string{"GENAI_OUTPUT"},
			},
			&cli.StringFlag{
				Name:    "llm-provider",
				Usage:   "LLM provider (available: openai, openrouter, mistral)",
				EnvVars: []string{"GENAI_LLM_PROVIDER"},
				Value:   string(openrouter.Name),
			},
			&cli.StringFlag{
				Name:    "llm-base-url",
				Usage:   "LLM provider base URL",
				EnvVars: []string{"GENAI_LLM_BASE_URL"},
				Value:   "https://openrouter.ai/api/v1",
			},
			&cli.StringFlag{
				Name:    "llm-api-key",
				Usage:   "LLM provider API key",
				EnvVars: []string{"GENAI_LLM_API_KEY"},
				Value:   "",
			},
			&cli.StringFlag{
				Name:    "llm-model",
				Usage:   "LLM model",
				EnvVars: []string{"GENAI_LLM_MODEL"},
				Value:   "openai/gpt-oss-20b:free",
			},
		},
		Action: func(cliCtx *cli.Context) error {
			ctx := cliCtx.Context

			client, err := provider.Create(ctx, provider.WithChatCompletionOptions(provider.ClientOptions{
				Provider: provider.Name(cliCtx.String("llm-provider")),
				BaseURL:  cliCtx.String("llm-base-url"),
				APIKey:   cliCtx.String("llm-api-key"),
				Model:    cliCtx.String("llm-model"),
			}))
			if err != nil {
				return errors.Wrap(err, "failed to create LLM client")
			}

			// Build messages
			messages, err := buildMessages(cliCtx)
			if err != nil {
				return errors.Wrap(err, "failed to build messages")
			}

			// Build chat completion options
			opts := []llm.ChatCompletionOptionFunc{
				llm.WithMessages(messages...),
				llm.WithTemperature(cliCtx.Float64("temperature")),
			}

			// Add JSON schema if provided
			if schemaPath := cliCtx.String("schema"); schemaPath != "" {
				schema, err := loadJSONSchema(schemaPath)
				if err != nil {
					return errors.Wrap(err, "failed to load JSON schema")
				}
				opts = append(opts, llm.WithJSONResponse(llm.NewResponseSchema(
					"response",
					"Structured response according to provided schema",
					schema,
				)))
			}

			// Generate completion
			response, err := client.ChatCompletion(ctx, opts...)
			if err != nil {
				return errors.Wrap(err, "failed to generate completion")
			}

			// Output the response
			content := response.Message().Content()
			if outputPath := cliCtx.String("output"); outputPath != "" {
				err := os.WriteFile(outputPath, []byte(content), 0644)
				if err != nil {
					return errors.Wrap(err, "failed to write output file")
				}
			} else {
				fmt.Print(content)
			}

			// Log usage information to stderr
			usage := response.Usage()
			slog.DebugContext(ctx, "Generation completed",
				"prompt_tokens", usage.PromptTokens(),
				"completion_tokens", usage.CompletionTokens(),
				"total_tokens", usage.TotalTokens(),
			)

			return nil
		},
	}
}

// buildMessages constructs the message list from CLI flags
func buildMessages(cliCtx *cli.Context) ([]llm.Message, error) {
	var messages []llm.Message

	// Add system message if provided
	if systemPrompt := cliCtx.String("system"); systemPrompt != "" {
		processedSystem, err := processPromptWithData(systemPrompt, cliCtx.String("system-data"))
		if err != nil {
			return nil, errors.Wrap(err, "failed to process system prompt")
		}
		messages = append(messages, llm.NewMessage(llm.RoleSystem, processedSystem))
	}

	// Process user prompt
	userPrompt := cliCtx.String("prompt")
	processedUser, err := processPromptWithData(userPrompt, cliCtx.String("prompt-data"))
	if err != nil {
		return nil, errors.Wrap(err, "failed to process user prompt")
	}

	// Handle file attachments
	files := cliCtx.StringSlice("file")
	if len(files) > 0 {
		attachments, err := processFileAttachments(files)
		if err != nil {
			return nil, errors.Wrap(err, "failed to process file attachments")
		}
		messages = append(messages, llm.NewMultimodalMessage(llm.RoleUser, processedUser, attachments...))
	} else {
		messages = append(messages, llm.NewMessage(llm.RoleUser, processedUser))
	}

	return messages, nil
}

// processPromptWithData processes a prompt template with optional JSON data
func processPromptWithData(promptText, dataInput string) (string, error) {

	// Check if promptText starts with "@" (file path)
	if strings.HasPrefix(promptText, "@") {
		filePath := promptText[1:] // Remove the "@" prefix
		content, err := os.ReadFile(filePath)
		if err != nil {
			return "", errors.Wrapf(err, "failed to read data file: %s", filePath)
		}
		promptText = string(content)
	}

	if dataInput == "" {
		return promptText, nil
	}

	var dataJSON string

	// Check if dataInput starts with "@" (file path)
	if strings.HasPrefix(dataInput, "@") {
		filePath := dataInput[1:] // Remove the "@" prefix
		content, err := os.ReadFile(filePath)
		if err != nil {
			return "", errors.Wrapf(err, "failed to read data file: %s", filePath)
		}
		dataJSON = string(content)
	} else {
		dataJSON = dataInput
	}

	var data interface{}
	if err := json.Unmarshal([]byte(dataJSON), &data); err != nil {
		return "", errors.Wrap(err, "failed to parse data JSON")
	}

	processed, err := prompt.Template(promptText, data)
	if err != nil {
		return "", errors.Wrap(err, "failed to process prompt template")
	}

	return processed, nil
}

// processFileAttachments processes file paths into attachments
func processFileAttachments(filePaths []string) ([]llm.Attachment, error) {
	var attachments []llm.Attachment

	for _, filePath := range filePaths {
		attachment, err := createAttachmentFromFile(filePath)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to process file: %s", filePath)
		}
		attachments = append(attachments, attachment)
	}

	return attachments, nil
}

// createAttachmentFromFile creates an attachment from a file path
func createAttachmentFromFile(filePath string) (llm.Attachment, error) {
	// Read file content
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read file")
	}

	// Determine MIME type
	mimeType := mime.TypeByExtension(filepath.Ext(filePath))
	if mimeType == "" {
		// Fallback to application/octet-stream for unknown types
		mimeType = "application/octet-stream"
	}

	// Encode content as base64
	encodedContent := base64.StdEncoding.EncodeToString(content)
	dataURL := fmt.Sprintf("data:%s;base64,%s", mimeType, encodedContent)

	// Determine attachment type based on MIME type
	var attachmentType llm.AttachmentType
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		attachmentType = llm.AttachmentTypeImage
	case strings.HasPrefix(mimeType, "audio/"):
		attachmentType = llm.AttachmentTypeAudio
	case strings.HasPrefix(mimeType, "video/"):
		attachmentType = llm.AttachmentTypeVideo
	default:
		attachmentType = llm.AttachmentTypeDocument
	}

	// Create attachment
	attachment, err := llm.NewBase64Attachment(attachmentType, mimeType, dataURL)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create attachment")
	}

	return attachment, nil
}

// loadJSONSchema loads and parses a JSON schema from a file
func loadJSONSchema(schemaPath string) (*jsonschema.Schema, error) {
	file, err := os.Open(schemaPath)
	if err != nil {
		return nil, errors.Wrap(err, "failed to open schema file")
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read schema file")
	}

	var schema jsonschema.Schema
	if err := json.Unmarshal(content, &schema); err != nil {
		return nil, errors.Wrap(err, "failed to parse JSON schema")
	}

	return &schema, nil
}

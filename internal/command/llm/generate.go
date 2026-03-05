package llm

import (
	"encoding/base64"
	"fmt"
	"log/slog"
	"mime"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/bornholm/genai/internal/command/common"
	"github.com/bornholm/genai/llm"
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
				Name:      "env-file",
				Usage:     "Environment file path",
				EnvVars:   []string{"GENAI_LLM_ENV_FILE"},
				Value:     ".env",
				TakesFile: true,
			},
			&cli.StringFlag{
				Name:    "env-prefix",
				Usage:   "Environment llm variables prefix",
				EnvVars: []string{"GENAI_LLM_ENV_PREFIX"},
				Value:   "GENAI_",
			},
			&cli.StringFlag{
				Name:    "output",
				Aliases: []string{"o"},
				Usage:   "Output file path (default: stdout)",
				EnvVars: []string{"GENAI_OUTPUT"},
			},
			&cli.IntFlag{
				Name:    "token-limit-chat-completion",
				Usage:   "Maximum tokens per minute for chat completion (0 to disable)",
				EnvVars: []string{"GENAI_TOKEN_LIMIT_CHAT_COMPLETION"},
				Value:   500000,
			},
			&cli.IntFlag{
				Name:    "token-limit-embeddings",
				Usage:   "Maximum tokens per minute for embeddings (0 to disable)",
				EnvVars: []string{"GENAI_TOKEN_LIMIT_EMBEDDINGS"},
				Value:   20000000,
			},
		},
		Action: func(cliCtx *cli.Context) error {
			ctx := cliCtx.Context

			envPrefix := cliCtx.String("env-prefix")
			envFile := cliCtx.String("env-file")

			// Build token limit options
			var tokenLimitOpts *common.TokenLimitOptions
			chatCompletionLimit := cliCtx.Int("token-limit-chat-completion")
			embeddingsLimit := cliCtx.Int("token-limit-embeddings")
			if chatCompletionLimit > 0 || embeddingsLimit > 0 {
				tokenLimitOpts = &common.TokenLimitOptions{
					ChatCompletionTokens:   chatCompletionLimit,
					ChatCompletionInterval: time.Minute,
					EmbeddingsTokens:       embeddingsLimit,
					EmbeddingsInterval:     time.Minute,
				}
			}

			client, err := common.NewResilientClient(ctx, envPrefix, envFile, tokenLimitOpts)
			if err != nil {
				return errors.Wrap(err, "failed to create llm client")
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

			responseSchema, err := common.GetResponseSchema(cliCtx, "schema")
			if err != nil {
				return errors.WithStack(err)
			}

			if responseSchema != nil {
				opts = append(opts, llm.WithJSONResponse(responseSchema))
			}

			// Generate completion
			response, err := client.ChatCompletion(ctx, opts...)
			if err != nil {
				return errors.Wrap(err, "failed to generate completion")
			}

			if err := common.WriteToOutput(cliCtx, "output", response.Message().Content(), false); err != nil {
				return errors.Wrap(err, "failed to write to output")
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
	if cliCtx.Count("system") != 0 {
		processedSystem, err := common.GetPrompt(cliCtx, "system", "system-data")
		if err != nil {
			return nil, errors.Wrap(err, "failed to process system prompt")
		}
		messages = append(messages, llm.NewMessage(llm.RoleSystem, processedSystem))
	}

	// Process user prompt
	processedUser, err := common.GetPrompt(cliCtx, "prompt", "prompt-data")
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

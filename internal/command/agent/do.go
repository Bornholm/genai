package agent

import (
	"encoding/base64"
	"log"
	"log/slog"
	"mime"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/agent/loop"
	"github.com/bornholm/genai/internal/command/common"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"

	// Import all provider implementations
	_ "github.com/bornholm/genai/llm/provider/all"
)

func Do() *cli.Command {
	return &cli.Command{
		Name:  "do",
		Usage: "Execute the given task with an agent",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:     "task",
				Usage:    "Description of the task to execute (text format, or @file to load from file)",
				EnvVars:  []string{"GENAI_TASK"},
				Required: true,
			},
			&cli.StringFlag{
				Name:    "task-data",
				Usage:   "Data to inject in the task prompt (JSON format, or @file to load from file)",
				EnvVars: []string{"GENAI_TASK_DATA"},
			},
			&cli.StringFlag{
				Name:    "additional-context",
				Usage:   "Additional context to pass to the agent (text format, or @file to load from file)",
				EnvVars: []string{"GENAI_ADDITIONAL_CONTEXT"},
			},
			&cli.StringFlag{
				Name:    "additional-context-data",
				Usage:   "Data to inject in the additional context (text format, or @file to load from file)",
				EnvVars: []string{"GENAI_ADDITIONAL_CONTEXT_DATA"},
			},
			&cli.StringFlag{
				Name:    "schema",
				Usage:   "JSON schema file path for structured response",
				EnvVars: []string{"GENAI_SCHEMA"},
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
			&cli.StringSliceFlag{
				Name:    "mcp",
				Usage:   "MCP server URL",
				EnvVars: []string{"GENAI_MCP"},
			},
			&cli.StringSliceFlag{
				Name:    "mcp-auth-token",
				Usage:   "MCP server auth token",
				EnvVars: []string{"GENAI_MCP_AUTH_TOKEN"},
			},
			&cli.StringFlag{
				Name:    "output",
				Aliases: []string{"o"},
				Usage:   "Output file path (default: stdout)",
				EnvVars: []string{"GENAI_OUTPUT"},
			},
			&cli.IntFlag{
				Name:    "max-iterations",
				Usage:   "Define the maximum number of iterations for the agent to make",
				Value:   100,
				EnvVars: []string{"GENAI_MAX_ITERATIONS"},
			},
			&cli.StringSliceFlag{
				Name:      "attachment",
				Aliases:   []string{"a"},
				Usage:     "File attachments to pass to the agent (supports images, documents, etc.)",
				EnvVars:   []string{"GENAI_ATTACHMENTS"},
				TakesFile: true,
			},
		},
		Action: func(cliCtx *cli.Context) error {
			ctx := cliCtx.Context

			envPrefix := cliCtx.String("env-prefix")
			envFile := cliCtx.String("env-file")

			client, err := common.NewResilientClient(ctx, envPrefix, envFile)
			if err != nil {
				return errors.Wrap(err, "failed to create llm client")
			}

			llmTools, close, err := common.GetMCPTools(cliCtx, "mcp", "mcp-auth-token")
			if err != nil {
				return errors.Wrap(err, "failed to get mcp tools")
			}

			defer close()

			if len(llmTools) > 0 {
				slog.DebugContext(ctx, "providing tools to agent", slog.Any("tools", slices.Collect(func(yield func(string) bool) {
					for _, t := range llmTools {
						if !yield(t.Name()) {
							return
						}
					}
				})))
			}

			taskPrompt, err := common.GetPrompt(cliCtx, "task", "task-data")
			if err != nil {
				return errors.Wrap(err, "failed to process task prompt")
			}

			additionalContext, err := common.GetPrompt(cliCtx, "additional-context", "additional-context-data")
			if err != nil {
				return errors.WithStack(err)
			}

			// Process attachments
			attachmentPaths := cliCtx.StringSlice("attachment")
			attachments, err := processAttachments(attachmentPaths)
			if err != nil {
				return errors.Wrap(err, "failed to process attachments")
			}

			// Build system prompt
			toolInfos := make([]loop.ToolInfo, len(llmTools))
			for i, t := range llmTools {
				toolInfos[i] = loop.ToolInfo{
					Name:        t.Name(),
					Description: t.Description(),
				}
			}

			systemPrompt, err := loop.DefaultSystemPrompt(toolInfos, additionalContext)
			if err != nil {
				return errors.Wrap(err, "failed to render system prompt")
			}

			// Create loop handler
			handler, err := loop.NewHandler(
				loop.WithClient(client),
				loop.WithTools(llmTools...),
				loop.WithSystemPrompt(systemPrompt),
				loop.WithMaxIterations(cliCtx.Int("max-iterations")),
			)
			if err != nil {
				return errors.Wrap(err, "failed to create handler")
			}

			// Create runner
			runner := agent.NewRunner(handler)

			// Run the agent
			var result string
			err = runner.Run(ctx, agent.NewInput(taskPrompt, attachments...), func(evt agent.Event) error {
				switch evt.Type() {
				case agent.EventTypeComplete:
					data := evt.Data().(*agent.CompleteData)
					result = data.Message
					slog.InfoContext(ctx, "agent completed", slog.String("message", data.Message))
				case agent.EventTypeToolCallStart:
					data := evt.Data().(*agent.ToolCallStartData)
					slog.DebugContext(ctx, "tool call started", slog.String("name", data.Name), slog.Any("params", data.Parameters))
				case agent.EventTypeToolCallDone:
					data := evt.Data().(*agent.ToolCallDoneData)
					slog.DebugContext(ctx, "tool call completed", slog.String("name", data.Name), slog.String("result", data.Result))
				case agent.EventTypeTodoUpdated:
					data := evt.Data().(*agent.TodoUpdatedData)
					slog.InfoContext(ctx, "todo list updated", slog.Any("items", data.Items))
				case agent.EventTypeError:
					data := evt.Data().(*agent.ErrorData)
					slog.ErrorContext(ctx, "agent error", slog.String("message", data.Message))
				}
				return nil
			})

			if err != nil {
				log.Fatalf("%+v", errors.WithStack(err))
			}

			if err := common.WriteToOutput(*cliCtx, "output", result); err != nil {
				return errors.Wrap(err, "failed to write to output")
			}

			return nil
		},
	}
}

// processAttachments reads files and converts them to llm.Attachment
func processAttachments(paths []string) ([]llm.Attachment, error) {
	attachments := make([]llm.Attachment, 0, len(paths))

	for _, path := range paths {
		attachment, err := fileToAttachment(path)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to process attachment '%s'", path)
		}
		attachments = append(attachments, attachment)
	}

	return attachments, nil
}

// fileToAttachment reads a file and creates an llm.Attachment
func fileToAttachment(path string) (llm.Attachment, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	mimeType := detectMimeType(path)
	attachmentType := detectAttachmentType(mimeType)

	base64Data := base64.StdEncoding.EncodeToString(data)

	attachment, err := llm.NewBase64Attachment(attachmentType, mimeType, base64Data)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return attachment, nil
}

// detectMimeType returns the MIME type based on file extension
func detectMimeType(path string) string {
	ext := filepath.Ext(path)
	mimeType := mime.TypeByExtension(ext)
	if mimeType == "" {
		// Default to octet-stream for unknown types
		return "application/octet-stream"
	}
	return strings.SplitN(mimeType, ";", 2)[0]
}

// detectAttachmentType returns the attachment type based on MIME type
func detectAttachmentType(mimeType string) llm.AttachmentType {
	if strings.HasPrefix(mimeType, "image/") {
		return llm.AttachmentTypeImage
	}
	if strings.HasPrefix(mimeType, "audio/") {
		return llm.AttachmentTypeAudio
	}
	if strings.HasPrefix(mimeType, "video/") {
		return llm.AttachmentTypeVideo
	}
	return llm.AttachmentTypeDocument
}

package agent

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"log/slog"
	"mime"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/bornholm/genai/a2a"
	"github.com/bornholm/genai/a2a/discovery"
	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/agent/loop"
	"github.com/bornholm/genai/internal/command/common"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"

	// Import all provider implementations
	_ "github.com/bornholm/genai/llm/provider/all"
	"github.com/bornholm/genai/llm/provider/openrouter"
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
			&cli.BoolFlag{
				Name:    "a2a",
				Usage:   "Enable A2A agent discovery to use discovered agents as tools",
				EnvVars: []string{"GENAI_A2A"},
				Value:   false,
			},
			&cli.BoolFlag{
				Name:    "no-planning",
				Usage:   "Disable the forced TodoWrite planning step at the start of the task (planning is enabled by default)",
				EnvVars: []string{"GENAI_NO_PLANNING"},
				Value:   false,
			},
			&cli.DurationFlag{
				Name:    "a2a-discovery-delay",
				Usage:   "Duration to wait for A2A agent discovery before starting the task",
				EnvVars: []string{"GENAI_A2A_DISCOVERY_DELAY"},
				Value:   5 * time.Second,
			},
			&cli.StringFlag{
				Name:    "reasoning-effort",
				Usage:   "Reasoning effort level: xhigh, high, medium, low, minimal, none (mutually exclusive with --reasoning-max-tokens)",
				EnvVars: []string{"GENAI_REASONING_EFFORT"},
				Value:   string(llm.ReasoningEffortMedium),
			},
			&cli.IntFlag{
				Name:    "reasoning-max-tokens",
				Usage:   "Maximum number of tokens to use for reasoning (mutually exclusive with --reasoning-effort)",
				EnvVars: []string{"GENAI_REASONING_MAX_TOKENS"},
				Value:   0,
			},
			&cli.BoolFlag{
				Name:    "unattended",
				Usage:   "Disable interactive UI output, fall back to slog for logging",
				EnvVars: []string{"GENAI_UNATTENDED"},
				Value:   false,
			},
			&cli.IntFlag{
				Name:    "max-tokens",
				Usage:   "Maximum number of tokens for context window management (use 0 for default: 80000)",
				EnvVars: []string{"GENAI_MAX_TOKENS"},
				Value:   0,
			},
			&cli.IntFlag{
				Name:    "max-tool-result-tokens",
				Usage:   "Maximum tokens per tool result to prevent context overflow (use 0 for default: 10000)",
				EnvVars: []string{"GENAI_MAX_TOOL_RESULT_TOKENS"},
				Value:   0,
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

			llmTools, close, err := common.GetMCPTools(cliCtx, "mcp", "mcp-auth-token")
			if err != nil {
				return errors.Wrap(err, "failed to get mcp tools")
			}

			defer close()

			// Create dynamic tool registry for A2A discovered agents
			dynamicRegistry := a2a.NewDynamicToolRegistry()

			// Start A2A discovery if enabled
			if cliCtx.Bool("a2a") {
				discoveryCtx, cancelDiscovery := context.WithCancel(ctx)
				defer cancelDiscovery()

				watcher := discovery.NewWatcher(&discovery.AgentEventHandlerFunc{
					OnDiscovered: func(agent *discovery.DiscoveredAgent) {
						tool := a2a.NewRemoteAgentTool(
							sanitizeToolName(agent.Name),
							fmt.Sprintf("Delegate tasks to the '%s' agent. %s", agent.Name, getAgentDescription(agent)),
							agent.URL,
						)
						dynamicRegistry.AddTool(tool)
						slog.Info("registered tool for discovered A2A agent", "name", agent.Name, "url", agent.URL)
					},
					OnRemoved: func(agent *discovery.DiscoveredAgent) {
						toolName := sanitizeToolName(agent.Name)
						dynamicRegistry.RemoveTool(toolName)
						slog.Info("unregistered tool for removed A2A agent", "name", agent.Name)
					},
				})

				go func() {
					if err := watcher.Watch(discoveryCtx); err != nil && discoveryCtx.Err() == nil {
						slog.Error("A2A discovery watcher error", "error", err)
					}
				}()

				// Wait for discovery to find agents
				discoveryDelay := cliCtx.Duration("a2a-discovery-delay")
				slog.Info("waiting for A2A agent discovery", "delay", discoveryDelay)
				time.Sleep(discoveryDelay)
			}

			// Combine MCP tools with discovered A2A agent tools
			allTools := append(llmTools, dynamicRegistry.ListTools()...)

			if len(allTools) > 0 {
				slog.DebugContext(ctx, "providing mcp tools to agent", slog.Any("tools", slices.Collect(func(yield func(string) bool) {
					for _, t := range allTools {
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
			toolInfos := make([]loop.ToolInfo, len(allTools))
			for i, t := range allTools {
				toolInfos[i] = loop.ToolInfo{
					Name:        t.Name(),
					Description: t.Description(),
				}
			}

			systemPrompt, err := loop.DefaultSystemPrompt(toolInfos, additionalContext)
			if err != nil {
				return errors.Wrap(err, "failed to render system prompt")
			}

			// Parse reasoning options
			reasoningOpts := common.GetReasoningOptions(cliCtx)

			// Create loop handler
			loopOpts := []loop.OptionFunc{
				loop.WithClient(client),
				loop.WithTools(allTools...),
				loop.WithSystemPrompt(systemPrompt),
				loop.WithMaxIterations(cliCtx.Int("max-iterations")),
				loop.WithForcePlanningStep(!cliCtx.Bool("no-planning")),
			}

			// Apply max-tokens if specified (0 means use default)
			maxTokens := cliCtx.Int("max-tokens")
			if maxTokens > 0 {
				loopOpts = append(loopOpts, loop.WithMaxTokens(maxTokens))
			}

			// Apply max-tool-result-tokens if specified (0 means use default)
			maxToolResultTokens := cliCtx.Int("max-tool-result-tokens")
			if maxToolResultTokens > 0 {
				loopOpts = append(loopOpts, loop.WithMaxToolResultTokens(maxToolResultTokens))
			}
			if reasoningOpts != nil {
				loopOpts = append(loopOpts, loop.WithReasoningOptions(reasoningOpts))
			}

			handler, err := loop.NewHandler(loopOpts...)
			if err != nil {
				return errors.Wrap(err, "failed to create handler")
			}

			// Create runner
			runner := agent.NewRunner(handler)

			ctx = openrouter.WithTransforms(ctx, []string{openrouter.TransformMiddleOut})

			// Determine if we should use UI mode
			unattended := cliCtx.Bool("unattended")

			// Run the agent
			var result string
			err = runner.Run(ctx, agent.NewInput(taskPrompt, attachments...), func(evt agent.Event) error {
				if unattended {
					// Use slog for logging in unattended mode
					switch evt.Type() {
					case agent.EventTypeComplete:
						data := evt.Data().(*agent.CompleteData)
						result = data.Message
						slog.InfoContext(ctx, "agent completed", slog.String("message", data.Message))
					case agent.EventTypeToolCallStart:
						data := evt.Data().(*agent.ToolCallStartData)
						slog.InfoContext(ctx, "tool call started", slog.String("name", data.Name), slog.Any("params", data.Parameters))
					case agent.EventTypeToolCallDone:
						data := evt.Data().(*agent.ToolCallDoneData)
						slog.DebugContext(ctx, "tool call completed", slog.String("name", data.Name), slog.String("result", data.Result))
					case agent.EventTypeTodoUpdated:
						data := evt.Data().(*agent.TodoUpdatedData)
						slog.InfoContext(ctx, "todo list updated", slog.Any("items", data.Items))
					case agent.EventTypeReasoning:
						data := evt.Data().(*agent.ReasoningData)
						slog.InfoContext(ctx, "reasoning tokens received",
							slog.String("reasoning", data.Reasoning),
							slog.Int("details", len(data.ReasoningDetails)),
						)
					case agent.EventTypeError:
						data := evt.Data().(*agent.ErrorData)
						slog.ErrorContext(ctx, "agent error", slog.String("message", data.Message))
					}
				} else {
					// Use lipgloss UI for output in interactive mode
					output := RenderEvent(evt)
					if output != "" {
						fmt.Println(output)
					}

					// Store result when complete
					if evt.Type() == agent.EventTypeComplete {
						data := evt.Data().(*agent.CompleteData)
						result = data.Message
					}
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

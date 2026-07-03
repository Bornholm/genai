package agent

import (
	"context"
	"encoding/base64"
	"fmt"
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
	agentconfig "github.com/bornholm/genai/internal/command/config"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/openrouter"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"
)

type doConfig struct {
	baseConfig

	a2aEnabled            bool
	a2aDiscoveryDelay     time.Duration
	task                  string
	taskData              any
	additionalContext     string
	additionalContextData any
	finalInstruction      string
	attachments           []string
	maxTokens             int
	maxToolResultTokens   int
	schema                string
	unattended            bool
	output                string
}

func loadDoConfig(cliCtx *cli.Context) (*doConfig, error) {
	if err := loadEnvFile(cliCtx); err != nil {
		return nil, errors.Wrap(err, "failed to load env file")
	}

	var yamlCfg *agentconfig.Config
	if configPath := cliCtx.String("config"); configPath != "" {
		cfg, err := agentconfig.Load(configPath)
		if err != nil {
			return nil, errors.Wrap(err, "failed to load config file")
		}
		yamlCfg = cfg
	}

	r := &cfgResolver{cliCtx: cliCtx, yamlCfg: yamlCfg}
	base := loadBaseConfig(r)

	cfg := &doConfig{baseConfig: base}

	cfg.task = r.string("task", func(c *agentconfig.Config) string {
		if c.Do != nil {
			return c.Do.Task
		}
		return ""
	})

	cfg.taskData = r.data("task-data", func(c *agentconfig.Config) map[string]any {
		if c.Do != nil {
			return c.Do.TaskData
		}
		return nil
	})

	cfg.additionalContext = r.string("additional-context", func(c *agentconfig.Config) string {
		if c.Do != nil {
			return c.Do.AdditionalContext
		}
		return ""
	})

	cfg.additionalContextData = r.data("additional-context-data", func(c *agentconfig.Config) map[string]any {
		if c.Do != nil {
			return c.Do.AdditionalContextData
		}
		return nil
	})

	cfg.finalInstruction = r.string("final-instruction", func(c *agentconfig.Config) string {
		if c.Do != nil {
			return c.Do.FinalInstruction
		}
		return ""
	})

	cfg.attachments = r.strings("attachment", func(c *agentconfig.Config) []string {
		if c.Do != nil {
			return c.Do.Attachments
		}
		return nil
	})

	cfg.maxTokens = r.int("max-tokens", func(c *agentconfig.Config) int {
		if c.Do != nil {
			return c.Do.MaxTokens
		}
		return 0
	})

	cfg.maxToolResultTokens = r.int("max-tool-result-tokens", func(c *agentconfig.Config) int {
		if c.Do != nil {
			return c.Do.MaxToolResultTokens
		}
		return 0
	})

	cfg.schema = r.string("schema", func(c *agentconfig.Config) string {
		if c.Do != nil {
			return c.Do.Schema
		}
		return ""
	})

	cfg.unattended = r.bool("unattended", func(c *agentconfig.Config) bool {
		if c.Do != nil {
			return c.Do.Unattended
		}
		return false
	})

	cfg.output = r.string("output", func(c *agentconfig.Config) string {
		if c.Do != nil {
			return c.Do.Output
		}
		return ""
	})

	if cliCtx.IsSet("a2a") {
		cfg.a2aEnabled = cliCtx.Bool("a2a")
	} else if yamlCfg != nil && yamlCfg.Do != nil && yamlCfg.Do.A2ADiscovery != nil {
		cfg.a2aEnabled = yamlCfg.Do.A2ADiscovery.Enabled
	} else {
		cfg.a2aEnabled = cliCtx.Bool("a2a")
	}

	if cliCtx.IsSet("a2a-discovery-delay") {
		cfg.a2aDiscoveryDelay = cliCtx.Duration("a2a-discovery-delay")
	} else if yamlCfg != nil && yamlCfg.Do != nil && yamlCfg.Do.A2ADiscovery != nil && yamlCfg.Do.A2ADiscovery.Delay != "" {
		if d, err := time.ParseDuration(yamlCfg.Do.A2ADiscovery.Delay); err == nil {
			cfg.a2aDiscoveryDelay = d
		}
	} else {
		cfg.a2aDiscoveryDelay = cliCtx.Duration("a2a-discovery-delay")
	}

	return cfg, nil
}

func Do() *cli.Command {
	return &cli.Command{
		Name:  "do",
		Usage: "Execute the given task with an agent",
		Flags: common.DoFlags,
		Action: func(cliCtx *cli.Context) error {
			ctx := cliCtx.Context

			cfg, err := loadDoConfig(cliCtx)
			if err != nil {
				return errors.Wrap(err, "failed to load config")
			}

			var tokenLimitOpts *common.TokenLimitOptions
			if cfg.chatCompletionLimit > 0 || cfg.embeddingsLimit > 0 {
				tokenLimitOpts = &common.TokenLimitOptions{
					ChatCompletionTokens:   cfg.chatCompletionLimit,
					ChatCompletionInterval: time.Minute,
					EmbeddingsTokens:       cfg.embeddingsLimit,
					EmbeddingsInterval:     time.Minute,
				}
			}

			client, err := common.NewResilientClient(ctx, cfg.envPrefix, cfg.envFile, tokenLimitOpts)
			if err != nil {
				return errors.Wrap(err, "failed to create llm client")
			}

			llmTools, close, err := common.GetMCPToolsFromURLs(cliCtx.Context, cfg.mcpURLs, cfg.mcpAuthTokens)
			if err != nil {
				return errors.Wrap(err, "failed to get mcp tools")
			}
			defer close()

			dynamicRegistry := a2a.NewDynamicToolRegistry()

			if cfg.a2aEnabled {
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

				slog.Info("waiting for A2A agent discovery", "delay", cfg.a2aDiscoveryDelay)
				time.Sleep(cfg.a2aDiscoveryDelay)
			}

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

			taskPrompt, err := common.GetPromptWithData(cliCtx, cfg.task, cfg.taskData)
			if err != nil {
				return errors.Wrap(err, "failed to process task prompt")
			}

			additionalContext, err := common.GetPromptWithData(cliCtx, cfg.additionalContext, cfg.additionalContextData)
			if err != nil {
				return errors.WithStack(err)
			}

			finalInstruction, err := common.GetPromptWithData(cliCtx, cfg.finalInstruction, cfg.taskData)
			if err != nil {
				return errors.WithStack(err)
			}

			attachments, err := processAttachments(cfg.attachments)
			if err != nil {
				return errors.Wrap(err, "failed to process attachments")
			}

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

			reasoningOpts := getReasoningOptions(cfg.reasoningEffort, cfg.reasoningMaxTokens)

			loopOpts := []loop.OptionFunc{
				loop.WithClient(client),
				loop.WithTools(allTools...),
				loop.WithSystemPrompt(systemPrompt),
				loop.WithMaxIterations(cfg.maxIterations),
				loop.WithForcePlanningStep(!cfg.noPlanning),
			}

			if finalInstruction != "" {
				loopOpts = append(loopOpts, loop.WithFinalInstruction(finalInstruction))
			}

			if cfg.maxTokens > 0 {
				loopOpts = append(loopOpts, loop.WithMaxTokens(cfg.maxTokens))
			}

			if cfg.maxToolResultTokens > 0 {
				loopOpts = append(loopOpts, loop.WithMaxToolResultTokens(cfg.maxToolResultTokens))
			}
			if reasoningOpts != nil {
				loopOpts = append(loopOpts, loop.WithReasoningOptions(reasoningOpts))
			}

			responseSchema, err := common.GetResponseSchema(cliCtx, "schema")
			if err != nil {
				return errors.Wrap(err, "failed to load response schema")
			}
			if responseSchema != nil {
				loopOpts = append(loopOpts, loop.WithResponseSchema(responseSchema))
			}

			handler, err := loop.NewHandler(loopOpts...)
			if err != nil {
				return errors.Wrap(err, "failed to create handler")
			}

			runner := agent.NewRunner(handler)

			ctx = openrouter.WithTransforms(ctx, []string{openrouter.TransformMiddleOut})

			var result string
			var streamed bool
			err = runner.Run(ctx, agent.NewInput(taskPrompt, attachments...), func(evt agent.Event) error {
				if cfg.unattended {
					switch evt.Type() {
					case agent.EventTypeTextDelta:
						data := evt.Data().(*agent.TextDeltaData)
						result += data.Delta
					case agent.EventTypeComplete:
						data := evt.Data().(*agent.CompleteData)
						if data.Message != "" {
							result = data.Message
						}
						slog.InfoContext(ctx, "agent completed", slog.String("message", result))
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
					switch evt.Type() {
					case agent.EventTypeTextDelta:
						fmt.Print(RenderEvent(evt))
						data := evt.Data().(*agent.TextDeltaData)
						result += data.Delta
						streamed = true
					case agent.EventTypeComplete:
						data := evt.Data().(*agent.CompleteData)
						if data.Message != "" {
							result = data.Message
							// When the final message was not streamed live (e.g. the
							// budget-exceeded summary or a schema-synthesis result, both
							// produced by non-streaming calls), render it now so the
							// terminal is not left without any final output.
							if !streamed {
								fmt.Println(RenderEvent(evt))
							}
						}
					default:
						output := RenderEvent(evt)
						if output != "" {
							fmt.Println(output)
						}
					}
				}
				return nil
			})

			if err != nil {
				return errors.WithStack(err)
			}

			if cfg.output != "" {
				if err := common.WriteToOutputString(cfg.output, result, cfg.unattended); err != nil {
					return errors.Wrap(err, "failed to write to output")
				}
			} else if !streamed {
				if err := common.WriteToOutputString("", result, cfg.unattended); err != nil {
					return errors.Wrap(err, "failed to write to output")
				}
			}

			return nil
		},
	}
}

func processAttachments(paths []string) ([]llm.Attachment, error) {
	attachments := make([]llm.Attachment, 0, len(paths))

	for _, path := range paths {
		if path == "" {
			continue
		}
		attachment, err := fileToAttachment(path)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to process attachment '%s'", path)
		}
		attachments = append(attachments, attachment)
	}

	return attachments, nil
}

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

func detectMimeType(path string) string {
	ext := filepath.Ext(path)
	mimeType := mime.TypeByExtension(ext)
	if mimeType == "" {
		return "application/octet-stream"
	}
	return strings.SplitN(mimeType, ";", 2)[0]
}

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

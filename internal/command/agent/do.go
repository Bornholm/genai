package agent

import (
	"log"
	"log/slog"
	"net/url"
	"slices"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/agent/task"
	"github.com/bornholm/genai/internal/command/common"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/mcp"
	"github.com/bornholm/genai/mcp/stdio"
	"github.com/bornholm/go-x/slogx"
	"github.com/google/shlex"
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
			&cli.StringFlag{
				Name:    "output",
				Aliases: []string{"o"},
				Usage:   "Output file path (default: stdout)",
				EnvVars: []string{"GENAI_OUTPUT"},
			},
			&cli.IntFlag{
				Name:    "tool-max-iterations",
				Usage:   "Define the maximum number of iterations in a row for tool calls",
				Value:   3,
				EnvVars: []string{"GENAI_TOOL_MAX_ITERATIONS"},
			},
			&cli.IntFlag{
				Name:    "max-iterations",
				Usage:   "Define the maximum number of iterations for the agent to make",
				Value:   10,
				EnvVars: []string{"GENAI_MAX_ITERATIONS"},
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

			opts := make([]task.HandlerOptionFunc, 0)

			llmTools, close, err := getMCPTools(cliCtx, "mcp")
			if err != nil {
				return errors.Wrap(err, "failed to create llm client")
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
				opts = append(opts, task.WithDefaultTools(llmTools...))
			}

			taskAgent := agent.New(
				task.NewHandler(
					client,
					opts...,
				),
			)

			// Start running the agent
			if _, _, err := taskAgent.Start(ctx); err != nil {
				return errors.Wrap(err, "could not start agent")
			}

			defer taskAgent.Stop()

			taskPrompt, err := common.GetPrompt(cliCtx, "task", "task-data")
			if err != nil {
				return errors.Wrap(err, "failed to process task prompt")
			}

			taskCtx := ctx

			responseSchema, err := common.GetResponseSchema(cliCtx, "schema")
			if err != nil {
				return errors.WithStack(err)
			}
			if responseSchema != nil {
				taskCtx = task.WithContextSchema(ctx, responseSchema)
			}

			additionalContext, err := common.GetPrompt(cliCtx, "additional-context", "additional-context-data")
			if err != nil {
				return errors.WithStack(err)
			}

			if additionalContext != "" {
				taskCtx = task.WithAdditionalContext(ctx, additionalContext)
			}

			maxToolIterations := cliCtx.Int("tool-max-iterations")
			taskCtx = task.WithContextMaxToolIterations(taskCtx, maxToolIterations)

			maxIterations := cliCtx.Int("max-iterations")
			taskCtx = task.WithContextMaxIterations(taskCtx, maxIterations)

			result, err := task.Do(taskCtx, taskAgent, taskPrompt,
				task.WithOnThought(func(evt task.ThoughtEvent) error {
					slog.InfoContext(ctx, "agent thought", slog.String("thought", evt.Thought()), slog.Int("iteration", evt.Iteration()))
					return nil
				}),
			)
			if err != nil {
				log.Fatalf("%+v", errors.WithStack(err))
			}

			if err := common.WriteToOutput(*cliCtx, "output", result.Result()); err != nil {
				return errors.Wrap(err, "failed to write to output")
			}

			return nil
		},
	}
}

func getMCPTools(ctx *cli.Context, param string) ([]llm.Tool, func(), error) {
	mcpURLs := ctx.StringSlice(param)

	clients := make([]mcp.Client, 0)

	close := func() {
		for _, c := range clients {
			if err := c.Stop(); err != nil {
				slog.ErrorContext(ctx.Context, "could not stop mcp client", slogx.Error(err))
			}
		}
	}

	for _, u := range mcpURLs {
		if _, err := url.ParseRequestURI(u); err == nil {
			// TODO Handle HTTP MCP servers
		} else {
			command, err := shlex.Split(u)
			if err != nil {
				return nil, nil, errors.Wrapf(err, "could not parse mcp server command '%s'", u)
			}
			c := stdio.NewClient(command...)

			slog.DebugContext(ctx.Context, "starting mcp client", slog.String("command", u))

			if err := c.Start(ctx.Context); err != nil {
				return nil, nil, errors.Wrapf(err, "could not start mcp client '%s'", u)
			}

			slog.DebugContext(ctx.Context, "mcp client started", slog.String("command", u))

			clients = append(clients, c)
		}
	}

	tools := make([]llm.Tool, 0)

	for i, c := range clients {
		clientTools, err := c.GetTools(ctx.Context)
		if err != nil {
			return nil, nil, errors.Wrapf(err, "could not retrieve mcp server '%s' tools", mcpURLs[i])
		}

		tools = append(tools, clientTools...)
	}

	return tools, close, nil

}

package agent

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/bornholm/genai/a2a"
	"github.com/bornholm/genai/a2a/discovery"
	"github.com/bornholm/genai/agent/loop"
	"github.com/bornholm/genai/internal/command/common"
	"github.com/bornholm/genai/llm"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"

	// Import all provider implementations
	_ "github.com/bornholm/genai/llm/provider/all"
)

func A2A() *cli.Command {
	return &cli.Command{
		Name:  "a2a",
		Usage: "Start an agent exposed via the A2A protocol on the local network",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "address",
				Aliases: []string{"a"},
				Value:   ":0", // Random available port
				Usage:   "Address to listen on (host:port)",
				EnvVars: []string{"A2A_ADDRESS"},
			},
			&cli.StringFlag{
				Name:    "name",
				Aliases: []string{"n"},
				Value:   "genai-agent",
				Usage:   "Agent name (used for mDNS announcement and agent card)",
				EnvVars: []string{"A2A_AGENT_NAME"},
			},
			&cli.StringFlag{
				Name:    "description",
				Value:   "A GenAI autonomous agent",
				Usage:   "Agent description",
				EnvVars: []string{"A2A_AGENT_DESCRIPTION"},
			},
			&cli.BoolFlag{
				Name:    "no-mdns",
				Value:   false,
				Usage:   "Disable mDNS announcement",
				EnvVars: []string{"A2A_NO_MDNS"},
			},
			&cli.BoolFlag{
				Name:    "discover",
				Value:   false,
				Usage:   "Enable mDNS discovery of peer agents",
				EnvVars: []string{"A2A_DISCOVER"},
			},
			&cli.StringFlag{
				Name:    "skills",
				Usage:   "Agent skills as JSON array or @file.json to load from file",
				EnvVars: []string{"A2A_SKILLS"},
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
			&cli.IntFlag{
				Name:    "max-iterations",
				Usage:   "Define the maximum number of iterations for the agent to make",
				Value:   100,
				EnvVars: []string{"GENAI_MAX_ITERATIONS"},
			},
			&cli.BoolFlag{
				Name:    "no-planning",
				Usage:   "Disable the forced TodoWrite planning step at the start of each task",
				EnvVars: []string{"GENAI_NO_PLANNING"},
				Value:   false,
			},
			&cli.StringFlag{
				Name:    "reasoning-effort",
				Usage:   "Reasoning effort level: xhigh, high, medium, low, minimal, none (mutually exclusive with --reasoning-max-tokens)",
				EnvVars: []string{"GENAI_REASONING_EFFORT"},
			},
			&cli.IntFlag{
				Name:    "reasoning-max-tokens",
				Usage:   "Maximum number of tokens to use for reasoning (mutually exclusive with --reasoning-effort)",
				EnvVars: []string{"GENAI_REASONING_MAX_TOKENS"},
				Value:   0,
			},
			&cli.StringFlag{
				Name:    "uuid",
				Usage:   "Agent UUID (defaults to randomly generated)",
				EnvVars: []string{"A2A_UUID"},
			},
			&cli.StringSliceFlag{
				Name:    "ignore",
				Usage:   "UUIDs of agents to ignore (can be specified multiple times)",
				EnvVars: []string{"A2A_IGNORE"},
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
			ctx, cancel := signal.NotifyContext(cliCtx.Context, os.Interrupt, syscall.SIGTERM)
			defer cancel()

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

			// Build LLM client using the same infrastructure as "agent do"
			llmClient, err := common.NewResilientClient(ctx, envPrefix, envFile, tokenLimitOpts)
			if err != nil {
				return errors.Wrap(err, "failed to create LLM client")
			}

			// Load tools (MCP servers, built-in tools) same as "agent do"
			tools, cleanup, err := common.GetMCPTools(cliCtx, "mcp", "mcp-auth-token")
			if err != nil {
				return errors.Wrap(err, "failed to load tools")
			}
			defer cleanup()

			// Create dynamic tool registry for discovered agents
			dynamicRegistry := a2a.NewDynamicToolRegistry()

			// Build the agent card
			agentName := cliCtx.String("name")
			skills, err := parseSkills(cliCtx.String("skills"))
			if err != nil {
				return errors.Wrap(err, "failed to parse skills")
			}

			// Build system prompt with initial tools
			toolInfos := make([]loop.ToolInfo, len(tools))
			for i, t := range tools {
				toolInfos[i] = loop.ToolInfo{
					Name:        t.Name(),
					Description: t.Description(),
				}
			}

			systemPrompt, err := loop.DefaultSystemPrompt(toolInfos, "")
			if err != nil {
				return errors.Wrap(err, "failed to render system prompt")
			}

			// Create a dynamic tool provider that combines static and dynamic tools
			allToolsProvider := func() []llm.Tool {
				return append(tools, dynamicRegistry.ListTools()...)
			}

			// Parse reasoning options
			reasoningOpts := common.GetReasoningOptions(cliCtx)

			// Build loop options
			loopOpts := []loop.OptionFunc{
				loop.WithMaxIterations(cliCtx.Int("max-iterations")),
				loop.WithForcePlanningStep(!cliCtx.Bool("no-planning")),
			}
			if reasoningOpts != nil {
				loopOpts = append(loopOpts, loop.WithReasoningOptions(reasoningOpts))
			}

			// Create the task handler backed by the agent loop
			handler := a2a.NewAgentTaskHandler(
				llmClient,
				a2a.WithToolsProvider(allToolsProvider),
				a2a.WithSystemPrompt(systemPrompt),
				a2a.WithLoopOptions(loopOpts...),
			)

			// Listen first so we know the actual port
			listener, err := net.Listen("tcp4", cliCtx.String("address"))
			if err != nil {
				return errors.Wrap(err, "failed to listen")
			}

			host, portStr, _ := net.SplitHostPort(listener.Addr().String())
			port, _ := strconv.Atoi(portStr)

			card := a2a.AgentCard{
				Name:        agentName,
				Description: cliCtx.String("description"),
				URL:         fmt.Sprintf("http://%s:%d", host, port),
				Version:     "1.0.0",
				Capabilities: a2a.AgentCapabilities{
					Streaming:         true,
					PushNotifications: false,
					StateTransitions:  true,
				},
				Skills:             skills,
				DefaultInputModes:  []string{"text"},
				DefaultOutputModes: []string{"text"},
			}

			server := a2a.NewServer(card, handler)

			// Generate or use provided unique ID for this agent instance
			agentID := cliCtx.String("uuid")
			if agentID == "" {
				agentID = uuid.New().String()
			} else {
				// Validate the provided UUID
				if _, err := uuid.Parse(agentID); err != nil {
					return errors.Wrap(err, "invalid UUID format")
				}
			}

			// Build set of ignored UUIDs (self + manually ignored)
			ignoredUUIDs := make(map[string]bool)
			ignoredUUIDs[agentID] = true
			for _, id := range cliCtx.StringSlice("ignore") {
				if _, err := uuid.Parse(id); err != nil {
					return errors.Wrapf(err, "invalid ignore UUID format: %s", id)
				}
				ignoredUUIDs[id] = true
			}

			// Start mDNS announcement
			if !cliCtx.Bool("no-mdns") {
				announcer, err := discovery.NewAnnouncer(
					discovery.WithInstance(agentName),
					discovery.WithPort(port),
					discovery.WithTXTRecords(
						fmt.Sprintf("id=%s", agentID),
						"version=1.0.0",
						fmt.Sprintf("description=%s", cliCtx.String("description")),
					),
				)
				if err != nil {
					return errors.Wrap(err, "failed to start mDNS announcer")
				}
				defer announcer.Shutdown()
			}

			// Optionally start mDNS discovery of peer agents with dynamic tool registration
			if cliCtx.Bool("discover") {
				watcher := discovery.NewWatcher(&discovery.AgentEventHandlerFunc{
					OnDiscovered: func(discoveredAgent *discovery.DiscoveredAgent) {
						// Ignore self and manually ignored agents (by UUID)
						if ignoredUUIDs[discoveredAgent.ID] {
							slog.Debug("ignoring agent in discovery", "id", discoveredAgent.ID, "name", discoveredAgent.Name)
							return
						}

						// Create a tool for the discovered agent
						tool := a2a.NewRemoteAgentTool(
							sanitizeToolName(discoveredAgent.Name),
							fmt.Sprintf("Delegate tasks to the '%s' agent. %s", discoveredAgent.Name, getAgentDescription(discoveredAgent)),
							discoveredAgent.URL,
						)
						dynamicRegistry.AddTool(tool)
						slog.Info("registered tool for discovered agent", "name", discoveredAgent.Name, "id", discoveredAgent.ID, "url", discoveredAgent.URL)
					},
					OnRemoved: func(discoveredAgent *discovery.DiscoveredAgent) {
						// Ignore self and manually ignored agents (by UUID)
						if ignoredUUIDs[discoveredAgent.ID] {
							return
						}
						toolName := sanitizeToolName(discoveredAgent.Name)
						dynamicRegistry.RemoveTool(toolName)
						slog.Info("unregistered tool for removed agent", "name", discoveredAgent.Name, "id", discoveredAgent.ID)
					},
				})

				go func() {
					if err := watcher.Watch(ctx); err != nil && ctx.Err() == nil {
						slog.Error("mDNS watcher error", "error", err)
					}
				}()
			}

			slog.Info("A2A agent started",
				"address", listener.Addr().String(),
				"name", agentName,
				"id", agentID,
			)

			httpServer := &http.Server{Handler: server}

			go func() {
				<-ctx.Done()
				slog.Info("Shutting down A2A server...")
				httpServer.Close()
			}()

			if err := httpServer.Serve(listener); err != nil && err != http.ErrServerClosed {
				return errors.Wrap(err, "server error")
			}

			return nil
		},
	}
}

// sanitizeToolName converts an agent name to a valid tool name
func sanitizeToolName(name string) string {
	// Replace spaces and special characters with underscores
	result := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			return r
		}
		return '_'
	}, name)
	return "agent_" + result
}

// getAgentDescription extracts description from TXT records or returns a default
func getAgentDescription(agent *discovery.DiscoveredAgent) string {
	for _, txt := range agent.TXT {
		if strings.HasPrefix(txt, "description=") {
			return strings.TrimPrefix(txt, "description=")
		}
	}
	return fmt.Sprintf("Remote agent at %s", agent.URL)
}

// parseSkills parses skills from JSON string or loads from file if prefixed with @
func parseSkills(input string) ([]a2a.AgentSkill, error) {
	if input == "" {
		return []a2a.AgentSkill{}, nil
	}

	var jsonData string

	// Check if input starts with "@" (file path)
	if strings.HasPrefix(input, "@") {
		filePath := input[1:] // Remove the "@" prefix
		content, err := os.ReadFile(filePath)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to read skills file: %s", filePath)
		}
		jsonData = string(content)
	} else {
		jsonData = input
	}

	var skills []a2a.AgentSkill
	if err := json.Unmarshal([]byte(jsonData), &skills); err != nil {
		return nil, errors.Wrap(err, "failed to parse skills JSON")
	}

	return skills, nil
}

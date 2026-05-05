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
	agentconfig "github.com/bornholm/genai/internal/command/config"
	"github.com/bornholm/genai/llm"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"
)

type a2aConfig struct {
	baseConfig

	address     string
	name        string
	description string
	noMdns      bool
	discover    bool
	skills      []a2a.AgentSkill
	uuid        string
	ignore      []string
}

func loadA2AConfig(cliCtx *cli.Context) (*a2aConfig, error) {
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

	cfg := &a2aConfig{baseConfig: base}

	cfg.address = r.string("address", func(c *agentconfig.Config) string {
		if c.A2A != nil {
			return c.A2A.Address
		}
		return ""
	})

	cfg.name = r.string("name", func(c *agentconfig.Config) string {
		if c.A2A != nil {
			return c.A2A.Name
		}
		return ""
	})

	cfg.description = r.string("description", func(c *agentconfig.Config) string {
		if c.A2A != nil {
			return c.A2A.Description
		}
		return ""
	})

	cfg.noMdns = r.bool("no-mdns", func(c *agentconfig.Config) bool {
		if c.A2A != nil {
			return c.A2A.NoMdns
		}
		return false
	})

	cfg.discover = r.bool("discover", func(c *agentconfig.Config) bool {
		if c.A2A != nil {
			return c.A2A.Discover
		}
		return false
	})

	if cliCtx.IsSet("skills") {
		skills, err := parseSkills(cliCtx.String("skills"))
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse skills")
		}
		cfg.skills = skills
	} else if yamlCfg != nil && yamlCfg.A2A != nil && len(yamlCfg.A2A.Skills) > 0 {
		skills, err := convertSkills(yamlCfg.A2A.Skills)
		if err != nil {
			return nil, errors.Wrap(err, "failed to convert skills from YAML")
		}
		cfg.skills = skills
	} else {
		cfg.skills = []a2a.AgentSkill{}
	}

	cfg.uuid = r.string("uuid", func(c *agentconfig.Config) string {
		if c.A2A != nil {
			return c.A2A.UUID
		}
		return ""
	})

	cfg.ignore = r.strings("ignore", func(c *agentconfig.Config) []string {
		if c.A2A != nil {
			return c.A2A.Ignore
		}
		return nil
	})

	return cfg, nil
}

func A2A() *cli.Command {
	return &cli.Command{
		Name:  "a2a",
		Usage: "Start an agent exposed via the A2A protocol on the local network",
		Flags: common.A2AFlags,
		Action: func(cliCtx *cli.Context) error {
			ctx, cancel := signal.NotifyContext(cliCtx.Context, os.Interrupt, syscall.SIGTERM)
			defer cancel()

			cfg, err := loadA2AConfig(cliCtx)
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

			llmClient, err := common.NewResilientClient(ctx, cfg.envPrefix, cfg.envFile, tokenLimitOpts)
			if err != nil {
				return errors.Wrap(err, "failed to create LLM client")
			}

			tools, cleanup, err := common.GetMCPToolsFromURLs(ctx, cfg.mcpURLs, cfg.mcpAuthTokens)
			if err != nil {
				return errors.Wrap(err, "failed to load tools")
			}
			defer cleanup()

			dynamicRegistry := a2a.NewDynamicToolRegistry()

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

			allToolsProvider := func() []llm.Tool {
				return append(tools, dynamicRegistry.ListTools()...)
			}

			reasoningOpts := getReasoningOptions(cfg.reasoningEffort, cfg.reasoningMaxTokens)

			loopOpts := []loop.OptionFunc{
				loop.WithMaxIterations(cfg.maxIterations),
				loop.WithForcePlanningStep(!cfg.noPlanning),
			}
			if reasoningOpts != nil {
				loopOpts = append(loopOpts, loop.WithReasoningOptions(reasoningOpts))
			}

			handler := a2a.NewAgentTaskHandler(
				llmClient,
				a2a.WithToolsProvider(allToolsProvider),
				a2a.WithSystemPrompt(systemPrompt),
				a2a.WithLoopOptions(loopOpts...),
			)

			listener, err := net.Listen("tcp4", cfg.address)
			if err != nil {
				return errors.Wrap(err, "failed to listen")
			}

			host, portStr, _ := net.SplitHostPort(listener.Addr().String())
			port, _ := strconv.Atoi(portStr)

			card := a2a.AgentCard{
				Name:        cfg.name,
				Description: cfg.description,
				URL:         fmt.Sprintf("http://%s:%d", host, port),
				Version:     "1.0.0",
				Capabilities: a2a.AgentCapabilities{
					Streaming:         true,
					PushNotifications: false,
					StateTransitions:  true,
				},
				Skills:             cfg.skills,
				DefaultInputModes:  []string{"text"},
				DefaultOutputModes: []string{"text"},
			}

			server := a2a.NewServer(card, handler)

			agentID := cfg.uuid
			if agentID == "" {
				agentID = uuid.New().String()
			} else {
				if _, err := uuid.Parse(agentID); err != nil {
					return errors.Wrap(err, "invalid UUID format")
				}
			}

			ignoredUUIDs := make(map[string]bool)
			ignoredUUIDs[agentID] = true
			for _, id := range cfg.ignore {
				if _, err := uuid.Parse(id); err != nil {
					return errors.Wrapf(err, "invalid ignore UUID format: %s", id)
				}
				ignoredUUIDs[id] = true
			}

			if !cfg.noMdns {
				announcer, err := discovery.NewAnnouncer(
					discovery.WithInstance(cfg.name),
					discovery.WithPort(port),
					discovery.WithTXTRecords(
						fmt.Sprintf("id=%s", agentID),
						"version=1.0.0",
						fmt.Sprintf("description=%s", cfg.description),
					),
				)
				if err != nil {
					return errors.Wrap(err, "failed to start mDNS announcer")
				}
				defer announcer.Shutdown()
			}

			if cfg.discover {
				watcher := discovery.NewWatcher(&discovery.AgentEventHandlerFunc{
					OnDiscovered: func(discoveredAgent *discovery.DiscoveredAgent) {
						if ignoredUUIDs[discoveredAgent.ID] {
							slog.Debug("ignoring agent in discovery", "id", discoveredAgent.ID, "name", discoveredAgent.Name)
							return
						}

						tool := a2a.NewRemoteAgentTool(
							sanitizeToolName(discoveredAgent.Name),
							fmt.Sprintf("Delegate tasks to the '%s' agent. %s", discoveredAgent.Name, getAgentDescription(discoveredAgent)),
							discoveredAgent.URL,
						)
						dynamicRegistry.AddTool(tool)
						slog.Info("registered tool for discovered agent", "name", discoveredAgent.Name, "id", discoveredAgent.ID, "url", discoveredAgent.URL)
					},
					OnRemoved: func(discoveredAgent *discovery.DiscoveredAgent) {
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
				"name", cfg.name,
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

func sanitizeToolName(name string) string {
	result := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			return r
		}
		return '_'
	}, name)
	return "agent_" + result
}

func getAgentDescription(agent *discovery.DiscoveredAgent) string {
	for _, txt := range agent.TXT {
		if strings.HasPrefix(txt, "description=") {
			return strings.TrimPrefix(txt, "description=")
		}
	}
	return fmt.Sprintf("Remote agent at %s", agent.URL)
}

func parseSkills(input string) ([]a2a.AgentSkill, error) {
	if input == "" {
		return []a2a.AgentSkill{}, nil
	}

	var jsonData string

	if strings.HasPrefix(input, "@") {
		filePath := input[1:]
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

func convertSkills(skillsAny []any) ([]a2a.AgentSkill, error) {
	if len(skillsAny) == 0 {
		return []a2a.AgentSkill{}, nil
	}

	data, err := json.Marshal(skillsAny)
	if err != nil {
		return nil, errors.Wrap(err, "failed to marshal skills for conversion")
	}

	var skills []a2a.AgentSkill
	if err := json.Unmarshal(data, &skills); err != nil {
		return nil, errors.Wrap(err, "failed to unmarshal skills")
	}

	return skills, nil
}

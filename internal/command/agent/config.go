package agent

import (
	"os"

	"github.com/bornholm/genai/internal/command/config"
	"github.com/bornholm/genai/llm"
	"github.com/joho/godotenv"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"
)

// baseConfig regroupe les champs communs aux commandes do et a2a.
type baseConfig struct {
	envPrefix           string
	envFile             string
	chatCompletionLimit int
	embeddingsLimit     int
	mcpURLs             []string
	mcpAuthTokens       []string
	maxIterations       int
	noPlanning          bool
	reasoningEffort     string
	reasoningMaxTokens  int
}

// cfgResolver résout les valeurs selon la priorité : CLI > YAML > défaut CLI.
type cfgResolver struct {
	cliCtx  *cli.Context
	yamlCfg *config.Config
}

func (r *cfgResolver) string(flag string, getter func(*config.Config) string) string {
	if r.cliCtx.IsSet(flag) {
		return r.cliCtx.String(flag)
	}
	if r.yamlCfg != nil {
		if val := getter(r.yamlCfg); val != "" {
			return val
		}
	}
	return r.cliCtx.String(flag)
}

func (r *cfgResolver) int(flag string, getter func(*config.Config) int) int {
	if r.cliCtx.IsSet(flag) {
		return r.cliCtx.Int(flag)
	}
	if r.yamlCfg != nil {
		if val := getter(r.yamlCfg); val > 0 {
			return val
		}
	}
	return r.cliCtx.Int(flag)
}

func (r *cfgResolver) bool(flag string, getter func(*config.Config) bool) bool {
	if r.cliCtx.IsSet(flag) {
		return r.cliCtx.Bool(flag)
	}
	if r.yamlCfg != nil {
		return getter(r.yamlCfg)
	}
	return r.cliCtx.Bool(flag)
}

func (r *cfgResolver) strings(flag string, getter func(*config.Config) []string) []string {
	if r.cliCtx.IsSet(flag) {
		return r.cliCtx.StringSlice(flag)
	}
	if r.yamlCfg != nil {
		if vals := getter(r.yamlCfg); len(vals) > 0 {
			return vals
		}
	}
	return r.cliCtx.StringSlice(flag)
}

// loadEnvFile charge le fichier .env spécifié par le flag "env-file" dans l'environnement du processus,
// afin que les variables soient disponibles lors de l'interpolation de la configuration YAML.
func loadEnvFile(cliCtx *cli.Context) error {
	envFile := cliCtx.String("env-file")
	if envFile == "" {
		return nil
	}
	if err := godotenv.Load(envFile); err != nil && !errors.Is(err, os.ErrNotExist) {
		return errors.Wrap(err, "failed to load env file")
	}
	return nil
}

// loadBaseConfig charge les champs communs aux deux commandes.
func loadBaseConfig(r *cfgResolver) baseConfig {
	cfg := baseConfig{}

	cfg.envFile = r.string("env-file", func(c *config.Config) string {
		if c.LLM != nil {
			return c.LLM.EnvFile
		}
		return ""
	})
	cfg.envPrefix = r.string("env-prefix", func(c *config.Config) string {
		if c.LLM != nil {
			return c.LLM.EnvPrefix
		}
		return ""
	})
	cfg.chatCompletionLimit = r.int("token-limit-chat-completion", func(c *config.Config) int {
		if c.LLM != nil {
			return c.LLM.TokenLimitChatCompletion
		}
		return 0
	})
	cfg.embeddingsLimit = r.int("token-limit-embeddings", func(c *config.Config) int {
		if c.LLM != nil {
			return c.LLM.TokenLimitEmbeddings
		}
		return 0
	})

	if r.cliCtx.IsSet("mcp") {
		cfg.mcpURLs = r.cliCtx.StringSlice("mcp")
	} else if r.yamlCfg != nil && r.yamlCfg.Agent != nil {
		for _, m := range r.yamlCfg.Agent.MCP {
			cfg.mcpURLs = append(cfg.mcpURLs, m.URL)
		}
	} else {
		cfg.mcpURLs = r.cliCtx.StringSlice("mcp")
	}

	if r.cliCtx.IsSet("mcp-auth-token") {
		cfg.mcpAuthTokens = r.cliCtx.StringSlice("mcp-auth-token")
	} else if r.yamlCfg != nil && r.yamlCfg.Agent != nil {
		for _, m := range r.yamlCfg.Agent.MCP {
			cfg.mcpAuthTokens = append(cfg.mcpAuthTokens, m.AuthToken)
		}
	} else {
		cfg.mcpAuthTokens = r.cliCtx.StringSlice("mcp-auth-token")
	}

	cfg.maxIterations = r.int("max-iterations", func(c *config.Config) int {
		if c.Agent != nil {
			return c.Agent.MaxIterations
		}
		return 0
	})
	cfg.noPlanning = r.bool("no-planning", func(c *config.Config) bool {
		if c.Agent != nil {
			return c.Agent.NoPlanning
		}
		return false
	})
	cfg.reasoningEffort = r.string("reasoning-effort", func(c *config.Config) string {
		if c.Agent != nil {
			return c.Agent.ReasoningEffort
		}
		return ""
	})
	cfg.reasoningMaxTokens = r.int("reasoning-max-tokens", func(c *config.Config) int {
		if c.Agent != nil {
			return c.Agent.ReasoningMaxTokens
		}
		return 0
	})

	return cfg
}

// getReasoningOptions construit les options de raisonnement à partir des valeurs résolues.
func getReasoningOptions(effort string, maxTokens int) *llm.ReasoningOptions {
	opts := &llm.ReasoningOptions{}
	hasOption := false

	if maxTokens > 0 {
		opts.MaxTokens = &maxTokens
		hasOption = true
	}

	if effort != "" && effort != "none" {
		e := llm.ReasoningEffort(effort)
		opts.Effort = &e
		hasOption = true
	}

	if !hasOption {
		return nil
	}

	return opts
}

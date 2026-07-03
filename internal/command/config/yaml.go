package config

import (
	"os"

	"github.com/pkg/errors"
	"gopkg.in/yaml.v3"
)

type Config struct {
	LLM   *LLMConfig   `yaml:"llm"`
	Agent *AgentConfig `yaml:"agent"`
	Do    *DoConfig    `yaml:"do"`
	A2A   *A2AConfig   `yaml:"a2a"`
}

// LLMConfig regroupe la configuration du client LLM.
type LLMConfig struct {
	EnvFile                  string `yaml:"envFile"`
	EnvPrefix                string `yaml:"envPrefix"`
	TokenLimitChatCompletion int    `yaml:"tokenLimitChatCompletion"`
	TokenLimitEmbeddings     int    `yaml:"tokenLimitEmbeddings"`
}

// AgentConfig contient les paramètres partagés entre les commandes do et a2a.
type AgentConfig struct {
	MaxIterations      int        `yaml:"maxIterations"`
	NoPlanning         bool       `yaml:"noPlanning"`
	ReasoningEffort    string     `yaml:"reasoningEffort"`
	ReasoningMaxTokens int        `yaml:"reasoningMaxTokens"`
	MCP                []MCPEntry `yaml:"mcp"`
}

type MCPEntry struct {
	URL       string `yaml:"url"`
	AuthToken string `yaml:"authToken"`
}

// DoConfig contient les paramètres spécifiques à la commande agent do.
type DoConfig struct {
	Task                  string              `yaml:"task"`
	TaskData              map[string]any      `yaml:"taskData"`
	AdditionalContext     string              `yaml:"additionalContext"`
	AdditionalContextData map[string]any      `yaml:"additionalContextData"`
	FinalInstruction      string              `yaml:"finalInstruction"`
	Attachments           []string            `yaml:"attachments"`
	MaxTokens             int                 `yaml:"maxTokens"`
	MaxToolResultTokens   int                 `yaml:"maxToolResultTokens"`
	Schema                string              `yaml:"schema"`
	Unattended            bool                `yaml:"unattended"`
	Output                string              `yaml:"output"`
	A2ADiscovery          *A2ADiscoveryConfig `yaml:"a2aDiscovery"`
}

type A2ADiscoveryConfig struct {
	Enabled bool   `yaml:"enabled"`
	Delay   string `yaml:"delay"`
}

// A2AConfig contient les paramètres spécifiques au serveur agent a2a.
type A2AConfig struct {
	Address     string   `yaml:"address"`
	Name        string   `yaml:"name"`
	Description string   `yaml:"description"`
	NoMdns      bool     `yaml:"noMdns"`
	Discover    bool     `yaml:"discover"`
	Skills      []any    `yaml:"skills"`
	UUID        string   `yaml:"uuid"`
	Ignore      []string `yaml:"ignore"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read config file")
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, errors.Wrap(err, "failed to parse config file")
	}

	interpolateConfig(&cfg)

	return &cfg, nil
}

func interpolateConfig(cfg *Config) {
	if cfg.LLM != nil {
		cfg.LLM.EnvFile = Interpolate(cfg.LLM.EnvFile)
		cfg.LLM.EnvPrefix = Interpolate(cfg.LLM.EnvPrefix)
	}

	if cfg.Agent != nil {
		cfg.Agent.ReasoningEffort = Interpolate(cfg.Agent.ReasoningEffort)
		for i := range cfg.Agent.MCP {
			cfg.Agent.MCP[i].URL = Interpolate(cfg.Agent.MCP[i].URL)
			cfg.Agent.MCP[i].AuthToken = Interpolate(cfg.Agent.MCP[i].AuthToken)
		}
	}

	if cfg.Do != nil {
		cfg.Do.Schema = Interpolate(cfg.Do.Schema)
		cfg.Do.Task = Interpolate(cfg.Do.Task)
		cfg.Do.AdditionalContext = Interpolate(cfg.Do.AdditionalContext)
		cfg.Do.FinalInstruction = Interpolate(cfg.Do.FinalInstruction)
		cfg.Do.Output = Interpolate(cfg.Do.Output)
		for i := range cfg.Do.Attachments {
			cfg.Do.Attachments[i] = Interpolate(cfg.Do.Attachments[i])
		}
		if cfg.Do.TaskData != nil {
			cfg.Do.TaskData = interpolateMapAny(cfg.Do.TaskData)
		}
		if cfg.Do.AdditionalContextData != nil {
			cfg.Do.AdditionalContextData = interpolateMapAny(cfg.Do.AdditionalContextData)
		}
		if cfg.Do.A2ADiscovery != nil {
			cfg.Do.A2ADiscovery.Delay = Interpolate(cfg.Do.A2ADiscovery.Delay)
		}
	}

	if cfg.A2A != nil {
		cfg.A2A.Address = Interpolate(cfg.A2A.Address)
		cfg.A2A.Name = Interpolate(cfg.A2A.Name)
		cfg.A2A.Description = Interpolate(cfg.A2A.Description)
		cfg.A2A.UUID = Interpolate(cfg.A2A.UUID)
		for i := range cfg.A2A.Ignore {
			cfg.A2A.Ignore[i] = Interpolate(cfg.A2A.Ignore[i])
		}
	}
}

func interpolateMapAny(m map[string]any) map[string]any {
	result := make(map[string]any, len(m))
	for k, v := range m {
		result[k] = InterpolateAny(v)
	}
	return result
}

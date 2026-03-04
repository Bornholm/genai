package common

import (
	"time"

	"github.com/urfave/cli/v2"
)

// Default values for token limits
const (
	DefaultTokenLimitChatCompletion = 500000
	DefaultTokenLimitEmbeddings     = 20000000
	DefaultMaxIterations            = 100
)

// DoFlags contains all flags for the "agent do" command.
var DoFlags []cli.Flag

func init() {
	DoFlags = []cli.Flag{
		// Task prompt
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
		// Additional context
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
		// Attachments
		&cli.StringSliceFlag{
			Name:      "attachment",
			Aliases:   []string{"a"},
			Usage:     "File attachments to pass to the agent (supports images, documents, etc.)",
			EnvVars:   []string{"GENAI_ATTACHMENTS"},
			TakesFile: true,
		},
		// A2A specific
		&cli.BoolFlag{
			Name:    "a2a",
			Usage:   "Enable A2A agent discovery to use discovered agents as tools",
			EnvVars: []string{"GENAI_A2A"},
			Value:   false,
		},
		&cli.DurationFlag{
			Name:    "a2a-discovery-delay",
			Usage:   "Duration to wait for A2A agent discovery before starting the task",
			EnvVars: []string{"GENAI_A2A_DISCOVERY_DELAY"},
			Value:   5 * time.Second,
		},
		// Context window management
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
		// Reasoning - do command uses medium as default
		&cli.StringFlag{
			Name:    "reasoning-effort",
			Usage:   "Reasoning effort level: xhigh, high, medium, low, minimal, none (mutually exclusive with --reasoning-max-tokens)",
			EnvVars: []string{"GENAI_REASONING_EFFORT"},
			Value:   "medium",
		},
		&cli.IntFlag{
			Name:    "reasoning-max-tokens",
			Usage:   "Maximum number of tokens to use for reasoning (mutually exclusive with --reasoning-effort)",
			EnvVars: []string{"GENAI_REASONING_MAX_TOKENS"},
			Value:   0,
		},
	}
	DoFlags = append(DoFlags, LLMClientFlags...)
	DoFlags = append(DoFlags, MCPFlags...)
	DoFlags = append(DoFlags, AgentLoopFlags...)
	DoFlags = append(DoFlags, OutputFlags...)
	DoFlags = append(DoFlags, SchemaFlags...)
}

// GenerateFlags contains all flags for the "llm generate" command.
var GenerateFlags []cli.Flag

// ChatFlags contains all flags for the "llm chat" command.
var ChatFlags []cli.Flag

// A2AFlags contains flags specific to the A2A server command.
var A2AFlags []cli.Flag

func init() {
	// GenerateFlags
	GenerateFlags = []cli.Flag{
		// System prompt
		&cli.StringFlag{
			Name:    "system",
			Usage:   "System prompt (text format, or @file to load from file)",
			EnvVars: []string{"GENAI_SYSTEM_PROMPT"},
		},
		&cli.StringFlag{
			Name:    "system-data",
			Usage:   "Data to inject in the system prompt (JSON format, or @file to load from file)",
			EnvVars: []string{"GENAI_SYSTEM_DATA"},
		},
		// User prompt
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
		// Files
		&cli.StringSliceFlag{
			Name:    "file",
			Usage:   "File attachments (can be specified multiple times)",
			EnvVars: []string{"GENAI_FILES"},
		},
		// Temperature
		&cli.Float64Flag{
			Name:    "temperature",
			Usage:   "Temperature for generation (0.0 to 2.0)",
			EnvVars: []string{"GENAI_TEMPERATURE"},
			Value:   0.7,
		},
	}
	GenerateFlags = append(GenerateFlags, LLMClientFlags...)
	GenerateFlags = append(GenerateFlags, OutputFlags...)
	GenerateFlags = append(GenerateFlags, SchemaFlags...)

	// ChatFlags
	ChatFlags = []cli.Flag{
		// System prompt
		&cli.StringFlag{
			Name:    "system",
			Usage:   "System prompt (text format, or @file to load from file)",
			EnvVars: []string{"GENAI_SYSTEM_PROMPT"},
		},
		&cli.StringFlag{
			Name:    "system-data",
			Usage:   "Data to inject in the system prompt (JSON format, or @file to load from file)",
			EnvVars: []string{"GENAI_SYSTEM_DATA"},
		},
		// Temperature
		&cli.Float64Flag{
			Name:    "temperature",
			Usage:   "Temperature for generation (0.0 to 2.0)",
			EnvVars: []string{"GENAI_TEMPERATURE"},
			Value:   0.4,
		},
	}
	ChatFlags = append(ChatFlags, LLMClientFlags...)
	ChatFlags = append(ChatFlags, MCPFlags...)
	ChatFlags = append(ChatFlags, ReasoningFlags...)

	// A2AFlags
	A2AFlags = []cli.Flag{
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
			Name:    "uuid",
			Usage:   "Agent UUID (defaults to randomly generated)",
			EnvVars: []string{"A2A_UUID"},
		},
		&cli.StringSliceFlag{
			Name:    "ignore",
			Usage:   "UUIDs of agents to ignore (can be specified multiple times)",
			EnvVars: []string{"A2A_IGNORE"},
		},
	}
	A2AFlags = append(A2AFlags, LLMClientFlags...)
	A2AFlags = append(A2AFlags, MCPFlags...)
	A2AFlags = append(A2AFlags, AgentLoopFlags...)
	A2AFlags = append(A2AFlags, ReasoningFlags...)
}

// LLMClientFlags returns the CLI flags for LLM client configuration.
// Include these in any command that needs to create an LLM client.
var LLMClientFlags = []cli.Flag{
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
	&cli.IntFlag{
		Name:    "token-limit-chat-completion",
		Usage:   "Maximum tokens per minute for chat completion (0 to disable)",
		EnvVars: []string{"GENAI_TOKEN_LIMIT_CHAT_COMPLETION"},
		Value:   DefaultTokenLimitChatCompletion,
	},
	&cli.IntFlag{
		Name:    "token-limit-embeddings",
		Usage:   "Maximum tokens per minute for embeddings (0 to disable)",
		EnvVars: []string{"GENAI_TOKEN_LIMIT_EMBEDDINGS"},
		Value:   DefaultTokenLimitEmbeddings,
	},
}

// MCPFlags returns the CLI flags for MCP server configuration.
// Include these in any command that supports MCP tools.
var MCPFlags = []cli.Flag{
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
}

// AgentLoopFlags returns the CLI flags for agent loop configuration.
// Include these in any command that runs an agent loop.
var AgentLoopFlags = []cli.Flag{
	&cli.IntFlag{
		Name:    "max-iterations",
		Usage:   "Define the maximum number of iterations for the agent to make",
		Value:   DefaultMaxIterations,
		EnvVars: []string{"GENAI_MAX_ITERATIONS"},
	},
	&cli.BoolFlag{
		Name:    "no-planning",
		Usage:   "Disable the forced TodoWrite planning step at the start of the task (planning is enabled by default)",
		EnvVars: []string{"GENAI_NO_PLANNING"},
		Value:   false,
	},
}

// OutputFlags returns the CLI flags for output configuration.
// Include these in any command that produces output.
var OutputFlags = []cli.Flag{
	&cli.StringFlag{
		Name:    "output",
		Aliases: []string{"o"},
		Usage:   "Output file path (default: stdout)",
		EnvVars: []string{"GENAI_OUTPUT"},
	},
}

// SchemaFlags returns the CLI flags for JSON schema configuration.
// Include these in any command that supports structured responses.
var SchemaFlags = []cli.Flag{
	&cli.StringFlag{
		Name:    "schema",
		Usage:   "JSON schema file path for structured response",
		EnvVars: []string{"GENAI_SCHEMA"},
	},
}

// PromptWithDataFlags returns CLI flags for a prompt with optional data injection.
// The promptParam and dataParam names should match the flag names used.
func PromptWithDataFlags(promptParam, dataParam string) []cli.Flag {
	return []cli.Flag{
		&cli.StringFlag{
			Name:     promptParam,
			Usage:    "Prompt text (text format, or @file to load from file)",
			Required: true,
			EnvVars:  []string{"GENAI_" + promptParam},
		},
		&cli.StringFlag{
			Name:    dataParam,
			Usage:   "Data to inject in the prompt (JSON format, or @file to load from file)",
			EnvVars: []string{"GENAI_" + dataParam},
		},
	}
}

// OptionalPromptWithDataFlags returns CLI flags for an optional prompt with optional data injection.
func OptionalPromptWithDataFlags(promptParam, dataParam string) []cli.Flag {
	return []cli.Flag{
		&cli.StringFlag{
			Name:    promptParam,
			Usage:   "Prompt text (text format, or @file to load from file)",
			EnvVars: []string{"GENAI_" + promptParam},
		},
		&cli.StringFlag{
			Name:    dataParam,
			Usage:   "Data to inject in the prompt (JSON format, or @file to load from file)",
			EnvVars: []string{"GENAI_" + dataParam},
		},
	}
}

// TokenLimitOptionsFromContext builds TokenLimitOptions from CLI context.
// Returns nil if both limits are set to 0 (disabled).
func TokenLimitOptionsFromContext(cliCtx *cli.Context) *TokenLimitOptions {
	chatCompletionLimit := cliCtx.Int("token-limit-chat-completion")
	embeddingsLimit := cliCtx.Int("token-limit-embeddings")

	if chatCompletionLimit == 0 && embeddingsLimit == 0 {
		return nil
	}

	return &TokenLimitOptions{
		ChatCompletionTokens:   chatCompletionLimit,
		ChatCompletionInterval: time.Minute,
		EmbeddingsTokens:       embeddingsLimit,
		EmbeddingsInterval:     time.Minute,
	}
}

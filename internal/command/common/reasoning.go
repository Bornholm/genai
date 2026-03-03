package common

import (
	"github.com/bornholm/genai/llm"
	"github.com/urfave/cli/v2"
)

// ReasoningFlags returns the CLI flags for reasoning configuration.
// Include these in any command that should support reasoning tokens.
var ReasoningFlags = []cli.Flag{
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
}

// GetReasoningOptions reads reasoning configuration from CLI context.
// Returns nil when no reasoning flags are set (disables reasoning).
func GetReasoningOptions(cliCtx *cli.Context) *llm.ReasoningOptions {
	effort := cliCtx.String("reasoning-effort")
	maxTokens := cliCtx.Int("reasoning-max-tokens")

	if effort == "" && maxTokens == 0 {
		return nil
	}

	opts := &llm.ReasoningOptions{}

	if effort != "" {
		e := llm.ReasoningEffort(effort)
		opts.Effort = &e
	}

	if maxTokens > 0 {
		opts.MaxTokens = &maxTokens
	}

	return opts
}

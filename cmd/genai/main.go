package main

import (
	"github.com/bornholm/genai/internal/command"
	"github.com/bornholm/genai/internal/command/agent"
	"github.com/bornholm/genai/internal/command/llm"
	proxyCommand "github.com/bornholm/genai/internal/command/proxy"

	// Import all provider implementations
	_ "github.com/bornholm/genai/llm/provider/all"
)

var (
	version string = "dev"
)

func main() {
	command.Main(
		"genai", version, "Generative AI command-line toolkit",
		llm.Root(),
		agent.Root(),
		proxyCommand.Root(),
	)
}

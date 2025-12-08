package main

import (
	"github.com/bornholm/genai/internal/command"
	"github.com/bornholm/genai/internal/command/llm"
)

var (
	version string = "dev"
)

func main() {
	command.Main(
		"genai", version, "Generative AI command-line toolkit",
		llm.Root(),
	)
}

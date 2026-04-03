package main

import (
	"context"
	"flag"
	"log"

	"github.com/bornholm/genai/llm/provider"

	_ "github.com/bornholm/genai/llm/provider/all"
	"github.com/bornholm/genai/llm/provider/env"
)

var (
	envFile string = ".env"
)

func init() {
	flag.StringVar(&envFile, "env-file", envFile, "client configuration environment file")
}

func main() {
	flag.Parse()
	ctx := context.Background()

	client, err := provider.Create(ctx, env.With("GENAI_", envFile))
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	inputs := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning is a subset of artificial intelligence",
	}

	res, err := client.Embeddings(ctx, inputs)
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	embeddings := res.Embeddings()
	usage := res.Usage()

	log.Printf("[INFO] Generated %d embeddings, each with %d dimensions", len(embeddings), len(embeddings[0]))
	log.Printf("[INFO] Usage: prompt tokens=%d, total tokens=%d", usage.PromptTokens(), usage.TotalTokens())
}

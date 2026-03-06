package main

import (
	"context"
	"flag"
	"log"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"

	// Imports client implementations

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

	// Create a client with chat completion implementation
	client, err := provider.Create(ctx, env.With("GENAI_", envFile))
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	// Create our chat completion history
	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "You are an expert in story-telling."),
		llm.NewMessage(llm.RoleUser, "Please tell me a beautiful story."),
	}

	// The chat completion options will now be validated before sending
	res, err := client.ChatCompletion(ctx,
		llm.WithMessages(messages...),
		llm.WithTemperature(0.7), // This will be validated to be between 0 and 2
	)
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	log.Printf("[STORY] %s", res.Message().Content())
}

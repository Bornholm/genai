package main

import (
	"context"
	"log"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"

	// Imports client implementations
	_ "github.com/bornholm/genai/llm/provider/openai"
)

func main() {
	ctx := context.Background()

	// Use GENAI_* environment variables to create a context to initialize the client
	// Read and load the .env file to populate the environment
	// See llm/provider/options.go for the available environment variable names
	client, err := provider.Create(ctx, provider.WithEnv("GENAI_", ".env"))
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	// Create our chat completion history
	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "You are an expert in story-telling."),
		llm.NewMessage(llm.RoleUser, "Please tell me a beautiful story."),
	}

	res, err := client.ChatCompletion(ctx,
		llm.WithMessages(messages...),
	)
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	log.Printf("[STORY] %s", res.Message().Content())
}

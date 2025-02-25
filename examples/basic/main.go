package main

import (
	"context"
	"log"

	// Imports LLM provider implementations
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	_ "github.com/bornholm/genai/llm/provider/openai"
	_ "github.com/bornholm/genai/llm/provider/openrouter"
)

func main() {
	// Use LLM_* environment variables to create a context
	// to initialize the LLM client
	// See llm/provider/context.go for the available keys
	ctx := context.Background()

	client, err := provider.Create(ctx, provider.WithEnvironment("LLM_"))
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

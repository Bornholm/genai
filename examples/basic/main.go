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

	ctx := context.Background()

	// Use LLM_* environment variables to create a context
	// to initialize the LLM client
	client, err := provider.Create(ctx, provider.WithEnv("LLM_", ".env"))
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

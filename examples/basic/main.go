package main

import (
	"context"
	"log"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"

	// Imports client implementations
	"github.com/bornholm/genai/llm/provider/openai"
	_ "github.com/bornholm/genai/llm/provider/openrouter"
)

func main() {
	ctx := context.Background()

	// Create a client with chat completion implementation
	client, err := provider.Create(ctx, provider.WithChatCompletionOptions(provider.ClientOptions{
		Provider: openai.Name,
		BaseURL:  "https://api.openai.com/v1/",
		Model:    "gpt-4o-mini",
		APIKey:   "<your-api-key>",
	}))
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

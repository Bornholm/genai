package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"github.com/bornholm/genai/llm"
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

	// Create a basic client
	client, err := provider.Create(ctx, env.With("GENAI_", envFile))
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	stream, err := client.ChatCompletionStream(ctx,
		llm.WithMessages(
			llm.NewMessage(llm.RoleUser, "Write a short poem about Go programming"),
		),
		llm.WithTemperature(0.7),
	)
	if err != nil {
		log.Printf("Error starting stream: %v", err)
		return
	}

	fmt.Print("AI: ")

	for chunk := range stream {
		if chunk.Error() != nil {
			log.Printf("Stream error: %v", chunk.Error())
			continue
		}

		if delta := chunk.Delta(); delta != nil {
			content := delta.Content()
			fmt.Print(content)
		}

		if chunk.IsComplete() {
			fmt.Printf("\n\nStream completed!")
			if usage := chunk.Usage(); usage != nil {
				fmt.Printf(" Usage: %d total tokens\n", usage.TotalTokens())
			}
			break
		}
	}
}

type StreamingToolCall struct {
	Index      int
	ID         string
	Name       string
	Parameters string
}

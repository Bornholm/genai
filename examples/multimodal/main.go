package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"

	// Import provider implementations
	_ "embed"

	_ "github.com/bornholm/genai/llm/provider/all"
	"github.com/bornholm/genai/llm/provider/env"
)

//go:embed duck.jpg
var duck []byte

func main() {
	ctx := context.Background()

	client, err := provider.Create(ctx, env.With("GENAI_", ".env"))
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	imageAttachment, err := llm.NewImageAttachment(
		"image/png",
		fmt.Sprintf("data:image/jpg;base64,%s", base64.StdEncoding.EncodeToString(duck)),
		false,
	)
	if err != nil {
		log.Fatalf("[FATAL] Failed to create image attachment: %s", err)
	}

	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "You are a helpful assistant that can analyze images."),
		llm.NewMultimodalMessage(
			llm.RoleUser,
			"What do you see in this image?",
			imageAttachment,
		),
	}

	res1, err := client.ChatCompletion(ctx,
		llm.WithMessages(messages...),
		llm.WithTemperature(0.1),
	)
	if err != nil {
		log.Printf("[ERROR] request failed: %s", err)
	} else {
		log.Printf("[RESPONSE] %s", res1.Message().Content())
	}
}

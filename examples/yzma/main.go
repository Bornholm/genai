package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/yzma"
)

var (
	modelFile   = flag.String("model", "", "path to the GGUF model file")
	libPath     = flag.String("lib", os.Getenv("YZMA_LIB"), "path to the llama.cpp library directory")
	prompt      = flag.String("prompt", "Are you ready to go?", "the prompt to use for chat completion")
	stream      = flag.Bool("stream", false, "enable streaming mode")
	embed       = flag.Bool("embed", false, "run embeddings example instead of chat completion")
	temperature = flag.Float64("temp", 0.7, "sampling temperature")
	maxTokens   = flag.Int("max-tokens", 128, "maximum number of tokens to generate")
)

func main() {
	flag.Parse()

	if *modelFile == "" {
		fmt.Println("Usage: yzma -model <path-to-model.gguf> [options]")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		os.Exit(1)
	}

	ctx := context.Background()

	if *embed {
		runEmbeddings(ctx)
	} else if *stream {
		runStreamingChat(ctx)
	} else {
		runChat(ctx)
	}
}

func runChat(ctx context.Context) {
	fmt.Println("=== Chat Completion Example ===")

	// Create chat completion client
	client, err := yzma.NewChatCompletionClient(
		yzma.WithModelPath(*modelFile),
		yzma.WithLibPath(*libPath),
		yzma.WithTemperature(*temperature),
		yzma.WithPredictSize(*maxTokens),
	)
	if err != nil {
		log.Fatalf("[FATAL] Failed to create client: %v", err)
	}
	defer client.Close()

	// Create messages
	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "You are a helpful assistant."),
		llm.NewMessage(llm.RoleUser, *prompt),
	}

	fmt.Printf("User: %s\n", *prompt)
	fmt.Print("Assistant: ")

	// Generate response
	res, err := client.ChatCompletion(ctx,
		llm.WithMessages(messages...),
		llm.WithTemperature(*temperature),
		llm.WithMaxCompletionTokens(*maxTokens),
	)
	if err != nil {
		log.Fatalf("[FATAL] Failed to generate response: %v", err)
	}

	fmt.Println(res.Message().Content())
	fmt.Printf("\nUsage: %d total tokens\n", res.Usage().TotalTokens())
}

func runStreamingChat(ctx context.Context) {
	fmt.Println("=== Streaming Chat Completion Example ===")

	// Create chat completion client
	client, err := yzma.NewChatCompletionClient(
		yzma.WithModelPath(*modelFile),
		yzma.WithLibPath(*libPath),
		yzma.WithTemperature(*temperature),
		yzma.WithPredictSize(*maxTokens),
	)
	if err != nil {
		log.Fatalf("[FATAL] Failed to create client: %v", err)
	}
	defer client.Close()

	// Create messages
	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "You are a helpful assistant."),
		llm.NewMessage(llm.RoleUser, *prompt),
	}

	fmt.Printf("User: %s\n", *prompt)
	fmt.Print("Assistant: ")

	// Start streaming
	stream, err := client.ChatCompletionStream(ctx,
		llm.WithMessages(messages...),
		llm.WithTemperature(*temperature),
		llm.WithMaxCompletionTokens(*maxTokens),
	)
	if err != nil {
		log.Fatalf("[FATAL] Failed to start stream: %v", err)
	}

	for chunk := range stream {
		if chunk.Error() != nil {
			log.Printf("[ERROR] Stream error: %v", chunk.Error())
			continue
		}

		if delta := chunk.Delta(); delta != nil {
			fmt.Print(delta.Content())
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

func runEmbeddings(ctx context.Context) {
	fmt.Println("=== Embeddings Example ===")

	// Create embeddings client
	client, err := yzma.NewEmbeddingsClient(
		yzma.WithEmbeddingsModelPath(*modelFile),
		yzma.WithEmbeddingsLibPath(*libPath),
		yzma.WithEmbeddingsNormalize(true),
	)
	if err != nil {
		log.Fatalf("[FATAL] Failed to create client: %v", err)
	}
	defer client.Close()

	// Generate embeddings for multiple texts
	inputs := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is a subset of artificial intelligence.",
		"Go is a programming language designed at Google.",
	}

	fmt.Printf("Generating embeddings for %d texts...\n\n", len(inputs))

	res, err := client.Embeddings(ctx, inputs)
	if err != nil {
		log.Fatalf("[FATAL] Failed to generate embeddings: %v", err)
	}

	embeddings := res.Embeddings()
	for i, emb := range embeddings {
		fmt.Printf("Text %d: %q\n", i+1, inputs[i])
		fmt.Printf("  Embedding dimension: %d\n", len(emb))
		fmt.Printf("  First 5 values: %.6f %.6f %.6f %.6f %.6f\n", emb[0], emb[1], emb[2], emb[3], emb[4])
		fmt.Println()
	}

	fmt.Printf("Total tokens: %d\n", res.Usage().TotalTokens())

	// Calculate similarity between first two embeddings
	if len(embeddings) >= 2 {
		similarity := cosineSimilarity(embeddings[0], embeddings[1])
		fmt.Printf("\nCosine similarity between text 1 and 2: %.6f\n", similarity)
	}
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrt(normA) * sqrt(normB))
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 100; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func init() {
	// Set default model path if not provided
	if *modelFile == "" {
		homeDir, err := os.UserHomeDir()
		if err == nil {
			*modelFile = filepath.Join(homeDir, ".yzma", "models", "SmolLM2-135M.Q4_K_M.gguf")
		}
	}
}

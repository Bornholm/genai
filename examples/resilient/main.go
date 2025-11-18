package main

import (
	"context"
	"log"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/circuitbreaker"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/ratelimit"
	"github.com/bornholm/genai/llm/retry"

	// Imports client implementations

	"github.com/bornholm/genai/llm/provider/env"
	_ "github.com/bornholm/genai/llm/provider/openrouter"
)

func main() {
	ctx := context.Background()

	// Create a basic client
	baseClient, err := provider.Create(ctx, env.With("GENAI_", ".env"))
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	// Wrap with retry logic (3 retries with 1 second base delay)
	retryClient := retry.Wrap(baseClient, time.Second, 3)

	// Wrap with rate limiting (max 10 requests per minute)
	rateLimitedClient := ratelimit.Wrap(retryClient, time.Minute/10, 1)

	// Wrap with circuit breaker (max 5 failures, 30 second reset timeout)
	resilientClient := circuitbreaker.NewClient(rateLimitedClient, 5, 30*time.Second)

	// Create our chat completion history
	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "You are an expert in story-telling."),
		llm.NewMessage(llm.RoleUser, "Please tell me a beautiful story."),
	}

	// This request will be protected by:
	// 1. Input validation
	// 2. Circuit breaker (fails fast if service is down)
	// 3. Rate limiting (prevents overwhelming the API)
	// 4. Retry logic (handles transient failures)
	res, err := resilientClient.ChatCompletion(ctx,
		llm.WithMessages(messages...),
		llm.WithTemperature(0.7),
		llm.WithMaxCompletionTokens(1000),
	)
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	log.Printf("[STORY] %s", res.Message().Content())
}

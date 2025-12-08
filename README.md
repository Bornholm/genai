# GenAI

[![Go Reference](https://pkg.go.dev/badge/github.com/bornholm/genai.svg)](https://pkg.go.dev/github.com/bornholm/genai)

## Overview

GenAI is a Go library that abstracts away the complexity of working with multiple AI service providers. It offers a unified interface for chat completions and other generative AI capabilities, allowing you to seamlessly switch between providers without changing your application code.

## Features

- Multi-provider support - Use OpenAI, OpenRouter, Mistral, Ollama and other providers with the same interface
- Unified API - Simple and consistent API for all providers
- Chat Completions - Create conversational AI experiences with ease
- Environment-based configuration - Configure your clients using environment variables
- Extensible - Easily add support for new providers or capabilities

## Supported Providers

- OpenAI
- OpenRouter
- Ollama
- Mistral

## Installation

```
go get github.com/Bornholm/genai
```

## Getting started

```go
ctx := context.Background()

// Create a client with chat completion implementation
client, err := provider.Create(ctx, provider.WithChatCompletionOptions(provider.ClientOptions{
  Provider: openai.Name,
  BaseURL:  "https://api.openai.com/v1/",
  Model:    "gpt-4o-mini",
  APIKey: "<your-api-key>",
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
```

## Examples

- [Basic](./examples/basic) - A basic example of a chat completion client with input validation
- [Using environment](./examples/environment) - An example of using environment variables to configure the client
- [Resilient](./examples/resilient) - An example showing how to build a resilient client with circuit breaker, rate limiting, and retry logic
- [Agent](./examples/agent) - An example of a [ReAct](https://arxiv.org/abs/2210.03629) agent with tools access
- [Multimodal](./examples/multimodal/) - An example of a multimodal LLM model call
- [JSON](./examples/json) - An example of a LLM call with structured output

## CLI

A CLI using this library is available. It supports the main operations provided by this library.

## License

[MIT](./LICENSE)

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

Here's how to get started with the GenAI library using environment variables for configuration:

```go
package main

import (
  "context"
  "flag"
  "log"

  "github.com/bornholm/genai/llm"
  "github.com/bornholm/genai/llm/provider"
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
```

Make sure to create a `.env` file with your API key. For example, with the Mistral provider:

```bash
GENAI_CHAT_COMPLETION_PROVIDER=mistral
GENAI_CHAT_COMPLETION_MISTRAL_BASE_URL=https://api.mistral.ai/v1/
GENAI_CHAT_COMPLETION_MISTRAL_API_KEY=<your_api_key>
GENAI_CHAT_COMPLETION_MISTRAL_MODEL=mistral-small-latest
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

## WASM

The GenAI library can be compiled to WebAssembly (WASM) to enable usage in browser-based environments, such as Thunderbird extensions or web applications. See [`WASM.md`](./WASM.md) for more informations.

### Tutorials

- [Getting started with `genai llm chat`](./docs/tutorials/getting-started-with-chat.md)

## License

[MIT](./LICENSE)

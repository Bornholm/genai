# Using genai with Yzma

Yzma is a Go library that provides bindings to llama.cpp for running local LLMs. The genai integration allows you to use Yzma as a provider for chat completion and embeddings.

## Prerequisites

- **GGUF model file**: A quantized LLM model in GGUF format (e.g., from Hugging Face)
- **llama.cpp library**: Compiled shared library for your platform
  - Download from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
  - Or build from source: `cmake -B build && cmake --build build`

## Installation

Le provider Yzma repose sur CGO et des liaisons natives vers llama.cpp. Il est désactivé par défaut et doit être activé explicitement via le build tag `yzma`.

### Build tag requis

Ajoutez le tag `yzma` à toutes vos commandes `go build` / `go run` / `go test` :

```bash
go build -tags yzma ./...
go run -tags yzma main.go
```

### Import

Pour charger tous les providers dont Yzma via le méta-package :

```go
import (
    _ "github.com/bornholm/genai/llm/provider/all" // nécessite -tags yzma pour inclure Yzma
)
```

Pour utiliser Yzma directement sans passer par le registre :

```go
import (
    "github.com/bornholm/genai/llm/provider/yzma"
)
```

## Quick Start

### Chat Completion

```go
package main

import (
    "context"
    "log"

    "github.com/bornholm/genai/llm"
    "github.com/bornholm/genai/llm/provider/yzma"
)

func main() {
    ctx := context.Background()

    client, err := yzma.NewChatCompletionClient(
        yzma.WithModelPath("/path/to/model.gguf"),
        yzma.WithLibPath("/path/to/llama.cpp/lib"),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    messages := []llm.Message{
        llm.NewMessage(llm.RoleSystem, "You are a helpful assistant."),
        llm.NewMessage(llm.RoleUser, "What is the capital of France?"),
    }

    resp, err := client.ChatCompletion(ctx, llm.WithMessages(messages...))
    if err != nil {
        log.Fatal(err)
    }

    println(resp.Message().Content())
}
```

### Streaming Chat Completion

```go
stream, err := client.ChatCompletionStream(ctx, llm.WithMessages(messages...))
if err != nil {
    log.Fatal(err)
}

for chunk := range stream {
    if delta := chunk.Delta(); delta != nil {
        print(delta.Content())
    }
    if chunk.IsComplete() {
        break
    }
}
```

### Embeddings

```go
client, err := yzma.NewEmbeddingsClient(
    yzma.WithEmbeddingsModelPath("/path/to/embedding-model.gguf"),
    yzma.WithEmbeddingsLibPath("/path/to/llama.cpp/lib"),
)
if err != nil {
    log.Fatal(err)
}
defer client.Close()

resp, err := client.Embeddings(ctx, []string{"Hello world", "Machine learning"})
if err != nil {
    log.Fatal(err)
}

embeddings := resp.Embeddings()
```

## Configuration Options

### Chat Completion Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `WithModelPath` | string | - | Path to GGUF model file |
| `WithModelURL` | string | - | URL to download model |
| `WithLibPath` | string | - | Path to llama.cpp library |
| `WithProcessor` | string | "auto" | Processing unit (cpu, cuda, metal) |
| `WithContextSize` | int | 40960 | Context window size |
| `WithTemperature` | float64 | 1.0 | Sampling temperature |
| `WithTopK` | int | 20 | Top-k sampling |
| `WithTopP` | float64 | 1.0 | Top-p sampling |
| `WithPredictSize` | int | 32768 | Max tokens to generate |

### Embeddings Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `WithEmbeddingsModelPath` | string | - | Path to GGUF model file |
| `WithEmbeddingsModelURL` | string | - | URL to download model |
| `WithEmbeddingsLibPath` | string | - | Path to llama.cpp library |
| `WithEmbeddingsProcessor` | string | "auto" | Processing unit |
| `WithEmbeddingsContextSize` | int | 40960 | Context window size |
| `WithEmbeddingsNormalize` | bool | true | Normalize output vectors |

## Environment Variables

You can also configure via environment variables:

```bash
export CHAT_COMPLETION_PROVIDER=yzma
export CHAT_COMPLETION_MODEL_PATH=/path/to/model.gguf
export CHAT_COMPLETION_LIB_PATH=/path/to/llama.cpp/lib
export CHAT_COMPLETION_TEMPERATURE=0.7
```

Then create client via the registry:

```go
client, err := provider.Create(ctx)
```

## Example

See `examples/yzma/main.go` for a complete example with CLI flags.

```bash
# Le build tag yzma est requis car le provider est désactivé par défaut
go run -tags yzma examples/yzma/main.go -model /path/to/model.gguf -lib /path/to/lib -prompt "Hello"
```

## Supported Models

Any GGUF-formatted model from Hugging Face should work. Popular options:

- **Small models** (< 500MB): Qwen2.5-0.5B, SmolLM2-135M
- **Medium models** (1-4GB): Qwen2.5-1.5B, Phi-3-mini
- **Larger models** (4GB+): Qwen2.5-7B, Llama-3-8B

Download models from [Hugging Face](https://huggingface.co/models?search=gguf) (filter by GGUF).
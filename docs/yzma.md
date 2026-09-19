# Using genai with Yzma

Yzma is a Go library that provides bindings to llama.cpp for running local LLMs. The genai integration allows you to use Yzma as a provider for chat completion and embeddings.

Yzma ships as a **provider plugin**: a separate binary, `genai-provider-yzma`, that the `genai` CLI (or any program using the `llm/provider/plugin` package) starts on demand. The main module stays free of the llama.cpp bindings. See [plugins/README.md](../plugins/README.md) for how plugins work.

## Prerequisites

- **GGUF model file**: A quantized LLM model in GGUF format (e.g., from Hugging Face)
- **llama.cpp library**: Compiled shared library for your platform
  - Download from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
  - Or build from source: `cmake -B build && cmake --build build`

## Installation

Build the plugin (no CGO required, yzma loads llama.cpp at runtime):

```bash
make build-plugins
# -> bin/genai-provider-yzma
```

Or take the `genai-provider-yzma` binary from its own release archive. The module carries
`replace` directives pointing at this repository, so `go install
github.com/bornholm/genai/plugins/yzma@latest` does not work: build it here,
or use a published binary.

## Quick Start with the CLI

```bash
export GENAI_PLUGIN_DIR=./bin
export GENAI_CHAT_COMPLETION_PROVIDER=yzma
export GENAI_CHAT_COMPLETION_YZMA_MODEL_PATH=/path/to/model.gguf
export GENAI_CHAT_COMPLETION_YZMA_LIB_PATH=/path/to/llama.cpp/lib
export GENAI_CHAT_COMPLETION_YZMA_TEMPERATURE=0.7

genai llm chat
```

Embeddings work the same way with the `GENAI_EMBEDDINGS_YZMA_*` prefix. Both capabilities share one plugin process.

## Using the provider in-process

The provider package can still be used directly by importing the plugin module, for programs that accept the llama.cpp dependency:

```go
package main

import (
    "context"
    "log"

    "github.com/bornholm/genai/llm"
    yzma "github.com/bornholm/genai/plugins/yzma/provider"
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

Importing the package also registers `yzma` in the provider registry, so `provider.Create` resolves it in-process instead of through the plugin.

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

Each option has an environment variable form, used through the plugin with the `GENAI_CHAT_COMPLETION_YZMA_` or `GENAI_EMBEDDINGS_YZMA_` prefix.

### Chat Completion Options

| Option | Variable | Type | Default | Description |
|--------|----------|------|---------|-------------|
| `WithModelPath` | `MODEL_PATH` | string | - | Path to GGUF model file |
| `WithModelURL` | `MODEL_URL` | string | - | URL to download model |
| `WithLibPath` | `LIB_PATH` | string | - | Path to llama.cpp library |
| `WithProcessor` | `PROCESSOR` | string | vide | Processing unit (cpu, cuda, metal); empty auto-detects at download |
| `WithContextSize` | `CONTEXT_SIZE` | int | 40960 | Context window size |
| `WithTemperature` | `TEMPERATURE` | float64 | 1.0 | Sampling temperature |
| `WithTopK` | `TOP_K` | int | 20 | Top-k sampling |
| `WithTopP` | `TOP_P` | float64 | 1.0 | Top-p sampling |
| `WithPredictSize` | `PREDICT_SIZE` | int | 32768 | Max tokens to generate |

### Embeddings Options

| Option | Variable | Type | Default | Description |
|--------|----------|------|---------|-------------|
| `WithEmbeddingsModelPath` | `MODEL_PATH` | string | - | Path to GGUF model file |
| `WithEmbeddingsModelURL` | `MODEL_URL` | string | - | URL to download model |
| `WithEmbeddingsLibPath` | `LIB_PATH` | string | - | Path to llama.cpp library |
| `WithEmbeddingsProcessor` | `PROCESSOR` | string | vide | Processing unit; empty auto-detects at download |
| `WithEmbeddingsContextSize` | `CONTEXT_SIZE` | int | 40960 | Context window size |
| `WithEmbeddingsNormalize` | `NORMALIZE` | bool | true | Normalize output vectors |

## Example

See `plugins/yzma/example/main.go` for a complete in-process example with CLI flags.

```bash
cd plugins/yzma
go run ./example -model /path/to/model.gguf -lib /path/to/lib -prompt "Hello"
```

## Supported Models

Any GGUF-formatted model from Hugging Face should work. Popular options:

- **Small models** (< 500MB): Qwen2.5-0.5B, SmolLM2-135M
- **Medium models** (1-4GB): Qwen2.5-1.5B, Phi-3-mini
- **Larger models** (4GB+): Qwen2.5-7B, Llama-3-8B

Download models from [Hugging Face](https://huggingface.co/models?search=gguf) (filter by GGUF).

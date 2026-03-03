# AGENT.md

This file provides guidance for AI coding agents working on this repository.

## Project Overview

`github.com/bornholm/genai` is a Go library and CLI toolkit for building LLM-powered agents, providing:

- A unified LLM client interface with pluggable providers (OpenAI, OpenRouter, Mistral)
- Resilience primitives: circuit breaker, rate limiting, retry, token limiting
- An agent loop with tool use, reasoning, and context management
- A2A (Agent-to-Agent) communication via mDNS discovery
- MCP (Model Context Protocol) client support
- A document extraction subsystem
- A CLI built with `urfave/cli/v2`

---

## Build & Test

```bash
# Build
go build ./...

# Run all tests
go test ./...

# Run a specific package
go test ./llm/...

# Run with race detector
go test -race ./...
```

> Integration tests that require external services (e.g., real LLM providers, Testcontainers)
> are gated by environment variables. Copy `.env.dist` to `.env` and fill in the required values.

## Repository Structure

```
genai/
├── agent/              # Agent loop, handler, middleware, events, context manager
│   ├── loop/           # Core agentic loop (context, truncation, etc.)
│   └── handler.go      # Handler interface and HandlerFunc adapter
├── a2a/                # Agent-to-Agent protocol
│   ├── discovery/      # mDNS announcer and browser
│   └── agent_card.go   # AgentCard public metadata document
├── extract/            # Document extraction (PDF → Markdown, etc.)
│   └── provider/       # Provider-specific extraction (mistral, …)
├── llm/                # Core LLM abstractions and provider implementations
│   ├── provider/       # OpenAI, OpenRouter, Mistral, registry, env loader
│   ├── circuitbreaker/ # Circuit-breaker decorator
│   ├── ratelimit/      # Request-level rate-limit decorator
│   ├── retry/          # Retry decorator
│   ├── tokenlimit/     # Token-level rate-limit decorator
│   ├── messageutil/    # Message helpers
│   ├── prompt/         # Prompt template engine
│   ├── context/        # LLM context helpers
│   └── hook/           # Hook/middleware client
├── mcp/                # Model Context Protocol clients (http, stdio)
├── text/               # Text utilities (hashing, word splitting)
├── internal/
│   └── command/common/ # Shared CLI helpers (prompt loading, output writing)
├── examples/           # Runnable examples
└── go.mod
```

## Coding Patterns & Conventions

### Error Handling

Always wrap errors with `github.com/pkg/errors` to preserve stack traces.
Never use `fmt.Errorf` for wrapping.

```go
// Correct
return nil, errors.WithStack(err)
return nil, errors.Wrap(err, "context message")
return nil, errors.Wrapf(err, "failed to open %s", path)

// Incorrect
return nil, fmt.Errorf("failed: %w", err)
```

### Interfaces & Adapters

- Define behaviour as small, focused interfaces.
- Provide a `Func` adapter for every single-method interface so callers can use plain functions.

```go
type Handler interface {
    Handle(ctx context.Context, input Input, emit EmitFunc) error
}

type HandlerFunc func(ctx context.Context, input Input, emit EmitFunc) error

func (fn HandlerFunc) Handle(ctx context.Context, input Input, emit EmitFunc) error {
    return fn(ctx, input, emit)
}
```

- Enforce interface satisfaction at compile time with blank-identifier assignments:

```go
var _ llm.Client = &Client{}
var _ json.Marshaler = jsonMarshaller{}
```

### Decorator / Middleware Pattern

Resilience wrappers (`circuitbreaker`, `ratelimit`, `retry`, `tokenlimit`) all follow the same decorator pattern:

```go
type Client struct {
    client llm.Client   // wrapped client
    // decorator-specific fields
}

func NewClient(client llm.Client, funcs ...OptionFunc) *Client { … }

var _ llm.Client = &Client{}
```

Apply the same pattern for any new cross-cutting concern.

### Functional Options

Configuration always uses the functional-options pattern with an `Options` struct,
an `OptionFunc` type, and a `NewOptions` constructor that applies defaults.

```go
type Options struct {
    Timeout time.Duration
}

type OptionFunc func(*Options)

func NewOptions(funcs ...OptionFunc) *Options {
    opts := &Options{Timeout: 30 * time.Second}
    for _, fn := range funcs {
        fn(opts)
    }
    return opts
}

func WithTimeout(d time.Duration) OptionFunc {
    return func(o *Options) { o.Timeout = d }
}
```

### Provider Registry

New LLM providers must register themselves in `llm/provider/registry.go` via an `init()` function,
and be imported by `llm/provider/all/import.go` for the "batteries-included" build tag.

```go
// llm/provider/myprovider/init.go
package myprovider

func init() {
    provider.Register("myprovider", factory)
}
```

### CLI Commands

- Commands are defined with `urfave/cli/v2`.
- Each command lives in its own file and exposes a constructor function returning `*cli.Command`.
- Shared flag helpers live in `internal/command/common/`.
- Use `common.GetPrompt` for loading prompt/data flags (supports `@filepath` syntax).
- Use `common.WriteToOutput` for writing results (honours `--output` flag or falls back to stdout).

### Concurrency

- Use `context.Context` for cancellation and deadlines; propagate it everywhere.
- When chunking token-bucket waits, split large `WaitN` calls into burst-sized chunks
  to avoid `"WaitN(n) exceeds limiter's burst"` errors (see `tokenlimit` package).
- Avoid goroutine leaks: always ensure goroutines have an exit path tied to context cancellation.

### Logging

Use the standard `log/slog` package. Pass loggers via `context` where needed (see `logx/context_handler.go`).

```go
slog.InfoContext(ctx, "action completed", "key", value)
```

### Testing

- Use table-driven tests with named `testCases` slices.
- Integration tests that need real services use `testcontainers-go`; gate them with `testing.Short()` or build tags.
- Compare fuzzy text output with Levenshtein distance (see `extract` tests) rather than exact equality.
- Use `embed.FS` for test fixtures (`testdata/` directories).

```go
//go:embed testdata
var myTestData embed.FS
```

## Adding a New LLM Provider

1. Create `llm/provider/<name>/`.
2. Implement `llm.Client` (ChatCompletion, Embeddings).
3. Add `init.go` that calls `provider.Register("<name>", factory)`.
4. Add a blank import to `llm/provider/all/import.go`.
5. Add environment variable documentation to `.env.dist`.
6. Write an integration test mirroring the pattern in existing providers.

## Adding a New Resilience Decorator

1. Create `llm/<concern>/client.go`.
2. Embed or delegate to `llm.Client`.
3. Follow the `NewClient(client llm.Client, funcs ...OptionFunc) *Client` constructor pattern.
4. Assert `var _ llm.Client = &Client{}`.
5. Add `options.go` with functional options if the decorator is configurable.

Do not add new dependencies without a clear justification; prefer the standard library.

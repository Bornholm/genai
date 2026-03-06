# AGENT.md

This file provides guidance for AI coding agents working on this repository.

## Commands

```bash
# Build the CLI binary
make build
# Output: bin/genai

# Run all tests
go test ./...

# Run a single test
go test ./agent/loop/... -run TestHandler_SimpleCompletion

# Run tests with verbose output
go test -v ./agent/...

# Build with CGO disabled (as Makefile does)
CGO_ENABLED=0 go build -o bin/genai ./cmd/genai
```

## Architecture

GenAI is a Go library (`github.com/bornholm/genai`) that provides a unified interface for multiple AI providers, plus a ReAct agent framework and an A2A (Agent-to-Agent) protocol implementation.

### Core LLM Layer (`llm/`)

The `llm.Client` interface composes three sub-interfaces: `ChatCompletionClient`, `ChatCompletionStreamingClient`, and `EmbeddingsClient`. Key types:

- `llm.Tool` / `llm.FuncTool` — tool definition and execution
- `llm.Attachment` — multimodal content (images, audio, video, documents)
- `llm.JSONSchema` — builder for JSON schema parameter definitions

### Provider System (`llm/provider/`)

Providers register themselves via `init()` functions using `provider.RegisterChatCompletion(name, factory)` and `provider.RegisterEmbeddings(name, factory)`. The global registry creates clients via `provider.Create(ctx, opts...)`.

Import `_ "github.com/bornholm/genai/llm/provider/all"` to load all providers at once. Supported providers: `openai`, `openrouter`, `ollama`, `mistral`.

Each provider's `ClientOptions` requires `Provider`, `BaseURL`, `Model`, and optionally `APIKey`. Environment variable prefixes: `CHAT_COMPLETION_PROVIDER`, `CHAT_COMPLETION_BASE_URL`, etc.

### Resilience Wrappers (`llm/circuitbreaker/`, `llm/ratelimit/`, `llm/retry/`)

Wrap any `llm.Client` with circuit breaker, rate limiting, or retry logic — these implement the same interfaces as the underlying clients.

### Agent Framework (`agent/`, `agent/loop/`)

The ReAct agent loop lives in `agent/loop/`. The main entry point is `loop.NewHandler(opts...)` which returns an `agent.Handler`. Key options:

- `loop.WithClient(client)` — the LLM client to use
- `loop.WithTools(tools...)` — tools available to the agent
- `loop.WithSystemPrompt(prompt)` — system prompt
- `loop.WithMaxIterations(n)` — iteration budget
- `loop.WithMaxTokens(n)` — context window limit (uses middle-out truncation)
- `loop.WithMaxToolResultTokens(n)` — truncate individual tool outputs
- `loop.WithForcePlanningStep(bool)` — enable/disable planning phase
- `loop.WithApprovalRequiredTools(names...)` + `loop.WithApprovalFunc(fn)` — human-in-the-loop approval

The `agent.Runner` wraps a `Handler` with optional `Middleware` and runs it synchronously. Events are emitted via `EmitFunc` of type `func(agent.Event) error`. Event types: `EventTypeComplete`, `EventTypeToolCallStart`, `EventTypeToolCallDone`, `EventTypeTodoUpdated`, `EventTypeReasoning`, `EventTypeError`.

The agent automatically includes `TodoRead`/`TodoWrite` tools for task tracking and emits `EventTypeTodoUpdated` when the todo list changes.

### A2A Protocol (`a2a/`)

Implements Google's Agent-to-Agent protocol (JSON-RPC over HTTP). Key components:

- `a2a.Client` — client for calling remote agents (tasks/send, tasks/sendSubscribe, tasks/get, tasks/cancel)
- `a2a.TaskHandler` interface — implement to serve an agent over A2A
- `a2a.AgentCard` — served at `/.well-known/agent.json` for discovery
- `a2a.DynamicToolRegistry` — runtime add/remove tools (used for A2A-discovered agents)
- `a2a/discovery/` — mDNS-based agent discovery via `discovery.NewWatcher`

Remote agents are exposed as `llm.Tool` instances via `a2a.NewRemoteAgentTool(name, description, url)`.

### MCP Integration (`mcp/`)

`mcp.Client` interface with `stdio` and `http` implementations. Converts MCP server tools into `[]llm.Tool` via `GetTools(ctx)`.

### Document Extraction (`extract/`)

Parallel to the LLM layer: `extract.Client` interface with provider registry and resilience wrappers. Supports Mistral and Marker providers.

### CLI (`cmd/genai/`, `internal/command/`)

Built with `urfave/cli/v2`. Top-level commands: `llm` (chat, embeddings), `agent` (do, a2a). The `agent do` command is the primary way to run an agent from the CLI with full MCP tool integration and A2A discovery.

### Adding a New Provider

1. Create package under `llm/provider/<name>/`
2. Register in `init()` via `provider.RegisterChatCompletion` and/or `provider.RegisterEmbeddings`
3. Add import to `llm/provider/all/import.go`

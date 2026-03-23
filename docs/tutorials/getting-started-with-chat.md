# Getting started with `genai llm chat`

This tutorial walks you through setting up the `genai` CLI for interactive chat with an LLM provider, including `.env` configuration and connecting an MCP server.

## Prerequisites

- `genai` binary installed (see `make build`, output: `bin/genai`)
- An API key for one of the supported providers: **OpenAI**, **Mistral**, **OpenRouter**, or a local **Ollama** instance

---

## Step 1 — Create your `.env` file

`genai` reads LLM provider credentials from an environment file (`.env` by default).

Create a `.env` file in your working directory:

```bash
touch .env
```

Then fill it with the appropriate variables for your provider.

### Option A — OpenAI

```dotenv
GENAI_CHAT_COMPLETION_PROVIDER=openai
GENAI_CHAT_COMPLETION_OPENAI_MODEL=gpt-4o
GENAI_CHAT_COMPLETION_OPENAI_API_KEY=sk-...
# Optional — override the default base URL
# GENAI_CHAT_COMPLETION_OPENAI_BASE_URL=https://api.openai.com/v1
```

### Option B — Mistral

```dotenv
GENAI_CHAT_COMPLETION_PROVIDER=mistral
GENAI_CHAT_COMPLETION_MISTRAL_MODEL=mistral-small-latest
GENAI_CHAT_COMPLETION_MISTRAL_API_KEY=<your-mistral-api-key>
# Optional — override the default base URL
# GENAI_CHAT_COMPLETION_MISTRAL_BASE_URL=https://api.mistral.ai/v1
```

### Option C — OpenRouter

```dotenv
GENAI_CHAT_COMPLETION_PROVIDER=openrouter
GENAI_CHAT_COMPLETION_OPENROUTER_MODEL=mistralai/ministral-8b
GENAI_CHAT_COMPLETION_OPENROUTER_API_KEY=sk-or-v1-...
# Optional — override the default base URL
# GENAI_CHAT_COMPLETION_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### Option D — Ollama (local), using OpenAI provider

```dotenv
GENAI_CHAT_COMPLETION_PROVIDER=openai
GENAI_CHAT_COMPLETION_OPENAI_MODEL=llama3.2
GENAI_CHAT_COMPLETION_OPENAI_BASE_URL=http://localhost:11434/v1
```

> **Note:** The `.env` file path can be changed with `--env-file <path>` or the `GENAI_LLM_ENV_FILE` environment variable.

---

## Step 2 — Start a chat session

Once your `.env` is ready, launch an interactive chat:

```bash
genai llm chat
```

You will be prompted to type messages. The conversation continues until you exit (e.g. `Ctrl+C` or `Ctrl+D`).

### Useful flags

| Flag                         | Description                                                            | Default      |
| ---------------------------- | ---------------------------------------------------------------------- | ------------ |
| `--system <text>`            | Set a system prompt                                                    | _(none)_     |
| `--system @<file>`           | Load system prompt from a file                                         | _(none)_     |
| `--temperature <float>`      | Sampling temperature (0.0–2.0)                                         | `0.4`        |
| `--reasoning-effort <level>` | Enable reasoning (`xhigh`, `high`, `medium`, `low`, `minimal`, `none`) | _(disabled)_ |
| `--env-file <path>`          | Path to the `.env` file                                                | `.env`       |

**Example with a system prompt and custom temperature:**

```bash
genai llm chat \
  --system "You are a senior Go developer. Be concise and precise." \
  --temperature 0.2
```

---

## Step 3 — Add an MCP server

MCP (Model Context Protocol) servers extend the LLM with additional tools — file access, issue trackers, knowledge bases, etc.

### HTTP MCP server

Pass the server URL with `--mcp`. If the server requires authentication, add `--mcp-auth-token`:

```bash
genai llm chat \
  --mcp "https://my-mcp-server.example.com/mcp" \
  --mcp-auth-token "<your-token>"
```

### stdio MCP server (command-based)

Some MCP servers run as local processes. Pass the full command as the `--mcp` value:

```bash
# Redmine MCP server via npx
genai llm chat \
  --mcp "npx -y @onozaty/redmine-mcp-server"

# Gitea MCP server via Go
genai llm chat \
  --mcp "go run gitea.com/gitea/gitea-mcp@latest -t stdio"
```

You can add multiple MCP servers by repeating the `--mcp` flag:

```bash
genai llm chat \
  --mcp "npx -y @onozaty/redmine-mcp-server" \
  --mcp "https://my-knowledge-base.example.com/mcp" \
  --mcp-auth-token "<token-for-knowledge-base>"
```

> **Order matters for auth tokens:** `--mcp-auth-token` values are matched positionally to `--mcp` values. Use an empty string `""` as a placeholder for servers that don't require a token.

# Provider plugins

A provider plugin is a standalone binary that serves one or more LLM
capabilities (chat completion, streaming, embeddings) to `genai` over
[hashicorp/go-plugin](https://github.com/hashicorp/go-plugin) (gRPC). It lets
a backend live outside the main module: heavy or native dependencies (yzma and
its llama.cpp bindings), unofficial APIs, or providers written in another
language.

## Using a plugin

The host looks for a binary named `genai-provider-<name>` in the directory
given by `--plugin-dir` / `GENAI_PLUGIN_DIR` (or `llm.pluginDir` in the YAML
config), and only there. Setting the directory is what enables plugins:
without it, an unknown provider name stays an error and no binary is run.
The `PATH` is never searched.

Selecting the provider works exactly like an in-tree one. Every variable under
the provider prefix is forwarded to the plugin, which validates it:

```bash
export GENAI_PLUGIN_DIR=./bin
export GENAI_CHAT_COMPLETION_PROVIDER=yzma
export GENAI_CHAT_COMPLETION_YZMA_MODEL_PATH=/models/qwen2.5-0.5b.gguf
export GENAI_CHAT_COMPLETION_YZMA_LIB_PATH=/opt/llama.cpp/lib
genai llm chat
```

`GENAI_<CAPABILITY>_<NAME>_COMMAND` names the binary for one capability,
bypassing the lookup. It still needs the plugin directory to be set: a
configuration file must not be enough to run a binary.

One process serves every capability of a plugin: configuring yzma for both
chat and embeddings starts a single binary. Processes are killed when the CLI
exits.

## Writing a plugin

The SDK is its own module, `github.com/bornholm/genai/plugin/sdk`. A plugin
wraps ordinary `llm` clients and hands their factories to `sdk.Serve`:

```go
package main

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/plugin/sdk"
)

type Options struct {
	BaseURL string `env:"BASE_URL"`
	APIKey  string `env:"API_KEY"`
	Model   string `env:"MODEL"`
}

func (o *Options) Validate() error {
	if o.APIKey == "" {
		return llm.NewValidationError("api_key", "API key is required")
	}
	return nil
}

func main() {
	sdk.Serve(sdk.Config{
		Name:    "acme",
		Version: "1.0.0",
		ChatCompletion: func(ctx context.Context, opts sdk.Options) (llm.ChatCompletionClient, error) {
			o := &Options{BaseURL: "https://api.acme.example"}
			if err := opts.Decode(o); err != nil {
				return nil, err
			}
			return acme.NewChatCompletionClient(o.BaseURL, o.APIKey, o.Model), nil
		},
	})
}
```

What crosses the process boundary:

- messages, attachments, tools (descriptors only: execution stays on the
  host), response schemas, reasoning options and extra fields;
- responses, usage (cache and cost counters included), stream chunks with
  reasoning and tool call deltas;
- typed errors: `llm.HTTPError`, `llm.ErrRateLimit`, `llm.ErrNoMessage`,
  `llm.ErrUnavailable` and `llm.ValidationError` are rebuilt on the host, so
  the retry and circuit breaker decorators behave as with an in-tree provider.

A client returned by a factory may implement
`llm.ChatCompletionStreamingClient`; when it does not, streaming requests are
answered from `ChatCompletion`.

Cancelling the host context cancels the gRPC stream, hence the context the
provider received: implement streaming with `llm.SendChunk` and
`llm.SendTerminalChunk`, as in-tree providers do.

## Plugins in this repository

| Plugin | Module | Capabilities |
|---|---|---|
| [`yzma`](./yzma) | `github.com/bornholm/genai/plugins/yzma` | chat completion, streaming, embeddings |

Build them with `make build-plugins`; the binaries land in `bin/`.

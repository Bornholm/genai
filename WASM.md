# Using GenAI with WebAssembly (WASM)

The GenAI library can be compiled to WebAssembly (WASM) to enable usage in browser-based environments, such as Thunderbird extensions or web applications.

## Prerequisites

- Go installed on your system (version 1.20 or higher recommended).
- A working Go environment with WASM support enabled.

## Example

```js
// 1. Load the WASM module (once at background script startup)
const go = new Go(); // from wasm_exec.js provided by Go

const result = await WebAssembly.instantiateStreaming(
  fetch("genai.wasm"),
  go.importObject
);

go.run(result.instance);

// 2. Create an LLM client
const client = genai.createClient({
  provider: "openai", // or "mistral", "openrouter"
  model: "gpt-4o",
  apiKey: "sk-...",
});

// 3. MCP Tools (optional)
const tools = await genai.createMCPTools("http://localhost:3000/sse");

// 4. Create the agent
const agent = genai.createAgent(client, {
  systemPrompt: "You are an email assistant in Thunderbird.",
  tools: tools,
  maxIterations: 20,
});

// 5. Run a task with streaming
await agent.run("Summarize this email: ...", (type, data) => {
  if (type === "text_delta") updateUI(data.delta);
  if (type === "tool_call_start") console.log("Tool call:", data.name);
  if (type === "complete") console.log("Complete:", data.message);
});
```

## API Reference

### `genai.createClient(config)`

Creates an LLM client with the specified configuration.

**Parameters:**

- `config.provider`: The LLM provider to use (`"openai"`, `"mistral"`, or `"openrouter"`).
- `config.model`: The model to use (e.g., `"gpt-4o"`).
- `config.baseURL`: Optional base URL for the API.
- `config.apiKey`: The API key for authentication.

### `genai.createMCPTools(endpoint)`

Connects to an MCP server (via SSE) and returns a handle to the available tools.

**Parameters:**

- endpoint: The URL of the MCP server's SSE endpoint.
  Returns:
- An object with a count property indicating the number of available tools.

### `genai.createAgent(client, config)`

Creates a ReAct agent with the specified client and configuration.

**Parameters:**

- client: The LLM client returned by genai.createClient.
- config.systemPrompt: The system prompt for the agent.
- config.tools: Optional array of tools (MCP or custom tools).
- config.maxIterations: Maximum number of iterations for the agent loop (default: 100).
- config.maxTokens: Maximum number of tokens for the context window (default: 80000).
- config.maxToolResultTokens: Maximum number of tokens for tool results (default: 10000).
- config.temperature: Temperature for the LLM's responses (optional).
- config.compressionRatio: Compression ratio for the context window (default: 0.0125).
- config.forcePlanningStep: Whether to force a planning step before the loop (default: false).
- config.reasoning: Configuration for reasoning mode (optional).
- config.approvalRequired: Array of tool names requiring approval (optional).
- config.approvalFunc: Function to approve tool execution (optional).
- config.tokenEstimator: Custom token estimator function (optional).

### `agent.run(message, onEvent, attachments)`

Runs the agent with the specified message and event handler.

**Parameters:**

- message: The input message for the agent.
- onEvent: A callback function to handle events (e.g., text_delta, tool_call_start, complete).
- attachments: Optional array of attachments (e.g., images, documents).
  Returns:
- An object with:
  - promise: A promise that resolves when the agent completes.
  - cancel: A function to cancel the agent's execution.

### `genai.registerTool(name, description, schemaJSON, callback)`

Registers a custom tool for use with agents.

**Parameters:**

- name: The name of the tool.
- description: A description of the tool.
- schemaJSON: The JSON schema for the tool's parameters.
- callback: A function to execute when the tool is called. The function receives the tool's arguments as a JSON object and must return a string or a Promise<string>.

**Example: Custom Tool Registration**

```js
genai.registerTool(
  "update_email",
  "Updates the body of the email being drafted.",
  JSON.stringify({
    type: "object",
    properties: {
      content: { type: "string", description: "The email body content" },
    },
    required: ["content"],
  }),
  async (args) => {
    await browser.compose.setComposeDetails(tabId, {
      plainTextBody: args.content,
    });
    return "Email updated successfully.";
  }
);
```

## Notes

- The wasm_exec.js file is required to run Go-compiled WASM modules. It can be obtained from the Go installation directory: $(go env GOROOT)/misc/wasm/wasm_exec.js.
- Ensure the genai.wasm file is served from a compatible location (e.g., the extension's resources directory).
- For security reasons, WASM modules in web environments are subject to CORS restrictions. Ensure your server is configured accordingly.

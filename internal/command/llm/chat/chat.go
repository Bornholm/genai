package chat

import (
	"context"
	"log/slog"
	"os"
	"slices"
	"strings"

	"github.com/bornholm/genai/internal/command/common"
	"github.com/bornholm/genai/llm"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"

	// Import all provider implementations
	_ "github.com/bornholm/genai/llm/provider/all"
)

func Chat() *cli.Command {
	return &cli.Command{
		Name:  "chat",
		Usage: "Start an interactive chat session",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:     "system",
				Usage:    "System prompt (text format, or @file to load from file)",
				EnvVars:  []string{"GENAI_SYSTEM_PROMPT"},
				Required: false,
			},
			&cli.StringFlag{
				Name:    "system-data",
				Usage:   "Data to inject in the system prompt (JSON format, or @file to load from file)",
				EnvVars: []string{"GENAI_SYSTEM_DATA"},
			},
			&cli.Float64Flag{
				Name:    "temperature",
				Usage:   "Temperature for generation (0.0 to 2.0)",
				EnvVars: []string{"GENAI_TEMPERATURE"},
				Value:   0.4,
			},
			&cli.StringFlag{
				Name:      "env-file",
				Usage:     "Environment file path",
				EnvVars:   []string{"GENAI_LLM_ENV_FILE"},
				Value:     ".env",
				TakesFile: true,
			},
			&cli.StringFlag{
				Name:    "env-prefix",
				Usage:   "Environment llm variables prefix",
				EnvVars: []string{"GENAI_LLM_ENV_PREFIX"},
				Value:   "GENAI_",
			},
			&cli.StringSliceFlag{
				Name:    "mcp",
				Usage:   "MCP server URL",
				EnvVars: []string{"GENAI_MCP"},
			},
			&cli.StringSliceFlag{
				Name:    "mcp-auth-token",
				Usage:   "MCP server auth token",
				EnvVars: []string{"GENAI_MCP_AUTH_TOKEN"},
			},
			&cli.StringFlag{
				Name:    "reasoning-effort",
				Usage:   "Reasoning effort level: xhigh, high, medium, low, minimal, none (mutually exclusive with --reasoning-max-tokens)",
				EnvVars: []string{"GENAI_REASONING_EFFORT"},
			},
			&cli.IntFlag{
				Name:    "reasoning-max-tokens",
				Usage:   "Maximum number of tokens to use for reasoning (mutually exclusive with --reasoning-effort)",
				EnvVars: []string{"GENAI_REASONING_MAX_TOKENS"},
				Value:   0,
			},
		},
		Action: func(cliCtx *cli.Context) error {
			ctx := cliCtx.Context

			envPrefix := cliCtx.String("env-prefix")
			envFile := cliCtx.String("env-file")

			client, err := common.NewResilientClient(ctx, envPrefix, envFile)
			if err != nil {
				return errors.Wrap(err, "failed to create llm client")
			}

			// Get MCP tools
			llmTools, close, err := common.GetMCPTools(cliCtx, "mcp", "mcp-auth-token")
			if err != nil {
				return errors.Wrap(err, "failed to get mcp tools")
			}

			defer close()

			if len(llmTools) > 0 {
				slog.DebugContext(ctx, "providing tools to chat", slog.Any("tools", slices.Collect(func(yield func(string) bool) {
					for _, t := range llmTools {
						if !yield(t.Name()) {
							return
						}
					}
				})))
			}

			// Get system prompt if provided
			var systemPrompt string
			if cliCtx.Count("system") != 0 {
				systemPrompt, err = common.GetPrompt(cliCtx, "system", "system-data")
				if err != nil {
					return errors.Wrap(err, "failed to process system prompt")
				}
			}

			temperature := cliCtx.Float64("temperature")

			// Get provider and model from environment
			providerName, modelName := getProviderAndModel(envPrefix)

			// Parse reasoning options
			reasoningOpts := common.GetReasoningOptions(cliCtx)

			// Create chat session
			chatSession := NewChatSession(
				client,
				WithSystemPrompt(systemPrompt),
				WithTemperature(temperature),
				WithTools(llmTools),
				WithProviderModel(providerName, modelName),
				WithReasoningOptions(reasoningOpts),
			)

			// Create and run UI
			ui, err := NewUI(chatSession)
			if err != nil {
				return errors.Wrap(err, "failed to create chat UI")
			}

			if err := ui.Run(ctx); err != nil {
				return errors.Wrap(err, "chat UI error")
			}

			return nil
		},
	}
}

// getProviderAndModel retrieves provider and model from environment variables
func getProviderAndModel(envPrefix string) (provider, model string) {
	// Try to get from environment with the given prefix
	providerEnv := envPrefix + "CHAT_COMPLETION_PROVIDER"
	modelEnv := envPrefix + "CHAT_COMPLETION_MODEL"

	provider = os.Getenv(providerEnv)
	model = os.Getenv(modelEnv)

	return provider, model
}

// ChatSession manages the conversation state
type ChatSession struct {
	client        llm.Client
	messages      []llm.Message
	systemPrompt  string
	temperature   float64
	tools         []llm.Tool
	provider      string
	model         string
	reasoning     *llm.ReasoningOptions
	onStreamChunk func(chunk string)
	onToolCall    func(name string, params map[string]any)
	onToolResult  func(name string, result string)
	onComplete    func()
	onError       func(err error)
}

// ChatSessionOptionFunc configures the chat session
type ChatSessionOptionFunc func(*ChatSession)

// WithSystemPrompt sets the system prompt
func WithSystemPrompt(prompt string) ChatSessionOptionFunc {
	return func(s *ChatSession) {
		s.systemPrompt = prompt
	}
}

// WithTemperature sets the temperature
func WithTemperature(temp float64) ChatSessionOptionFunc {
	return func(s *ChatSession) {
		s.temperature = temp
	}
}

// WithTools sets the available tools
func WithTools(tools []llm.Tool) ChatSessionOptionFunc {
	return func(s *ChatSession) {
		s.tools = tools
	}
}

// WithProviderModel sets the provider and model information
func WithProviderModel(provider, model string) ChatSessionOptionFunc {
	return func(s *ChatSession) {
		s.provider = provider
		s.model = model
	}
}

// WithReasoningOptions sets the reasoning options for the chat session.
// When set, reasoning tokens are requested from the model and preserved
// across multi-turn tool-call conversations.
func WithReasoningOptions(opts *llm.ReasoningOptions) ChatSessionOptionFunc {
	return func(s *ChatSession) {
		s.reasoning = opts
	}
}

// GetProviderModel returns the provider and model information
func (s *ChatSession) GetProviderModel() (provider, model string) {
	return s.provider, s.model
}

// OnStreamChunk sets the stream chunk callback
func OnStreamChunk(fn func(chunk string)) ChatSessionOptionFunc {
	return func(s *ChatSession) {
		s.onStreamChunk = fn
	}
}

// OnToolCall sets the tool call callback
func OnToolCall(fn func(name string, params map[string]any)) ChatSessionOptionFunc {
	return func(s *ChatSession) {
		s.onToolCall = fn
	}
}

// OnToolResult sets the tool result callback
func OnToolResult(fn func(name string, result string)) ChatSessionOptionFunc {
	return func(s *ChatSession) {
		s.onToolResult = fn
	}
}

// OnComplete sets the complete callback
func OnComplete(fn func()) ChatSessionOptionFunc {
	return func(s *ChatSession) {
		s.onComplete = fn
	}
}

// OnError sets the error callback
func OnError(fn func(err error)) ChatSessionOptionFunc {
	return func(s *ChatSession) {
		s.onError = fn
	}
}

// NewChatSession creates a new chat session
func NewChatSession(client llm.Client, opts ...ChatSessionOptionFunc) *ChatSession {
	session := &ChatSession{
		client:      client,
		messages:    make([]llm.Message, 0),
		temperature: 0.7,
	}
	for _, opt := range opts {
		opt(session)
	}
	return session
}

// SendMessage sends a message and streams the response
func (s *ChatSession) SendMessage(ctx context.Context, content string) error {
	// Add user message
	if content != "" {
		s.messages = append(s.messages, llm.NewMessage(llm.RoleUser, content))
	}

	// Build options
	completionOpts := []llm.ChatCompletionOptionFunc{
		llm.WithMessages(s.getMessages()...),
		llm.WithTemperature(s.temperature),
	}

	if len(s.tools) > 0 {
		completionOpts = append(completionOpts, llm.WithTools(s.tools...), llm.WithToolChoice(llm.ToolChoiceAuto))
	}

	if s.reasoning != nil {
		completionOpts = append(completionOpts, llm.WithReasoning(s.reasoning))
	}

	// Use streaming API for text content
	stream, err := s.client.ChatCompletionStream(ctx, completionOpts...)
	if err != nil {
		if s.onError != nil {
			s.onError(err)
		}
		return errors.WithStack(err)
	}

	var assistantContent strings.Builder
	var toolCalls []*streamingToolCall

	for chunk := range stream {
		if chunk.Error() != nil {
			if s.onError != nil {
				s.onError(chunk.Error())
			}
			return errors.WithStack(chunk.Error())
		}

		if chunk.IsComplete() {
			break
		}

		delta := chunk.Delta()
		if delta == nil {
			continue
		}

		// Handle text content - stream in real-time
		if delta.Content() != "" {
			assistantContent.WriteString(delta.Content())
			if s.onStreamChunk != nil {
				s.onStreamChunk(delta.Content())
			}
		}

		// Handle tool call deltas
		for _, tcDelta := range delta.ToolCalls() {
			// Find or create the tool call accumulator
			var found *streamingToolCall
			for _, tc := range toolCalls {
				// Match by ID if available, otherwise by index
				if tcDelta.ID() != "" && tc.id == tcDelta.ID() {
					found = tc
					break
				}
				if tcDelta.ID() == "" && tc.index == tcDelta.Index() {
					found = tc
					break
				}
			}

			if found == nil {
				found = &streamingToolCall{
					index: tcDelta.Index(),
					id:    tcDelta.ID(),
					name:  tcDelta.Name(),
				}
				toolCalls = append(toolCalls, found)
			}

			// Accumulate data
			if tcDelta.ID() != "" {
				found.id = tcDelta.ID()
			}
			if tcDelta.Name() != "" {
				found.name = tcDelta.Name()
			}
			found.parameters += tcDelta.ParametersDelta()
		}
	}

	// Handle tool calls if any - use non-streaming for tool execution loop
	if len(toolCalls) > 0 {
		// Ensure all tool calls have valid IDs
		for _, tc := range toolCalls {
			if tc.id == "" {
				tc.id = "tc_" + uuid.New().String()[:8]
			}
		}

		// Convert streaming tool calls to llm.ToolCall
		finalToolCalls := make([]llm.ToolCall, len(toolCalls))
		for i, tc := range toolCalls {
			finalToolCalls[i] = tc
		}

		// Add assistant message with tool calls
		s.messages = append(s.messages, llm.NewToolCallsMessage(finalToolCalls...))

		// Execute tool calls
		for _, tc := range toolCalls {
			if s.onToolCall != nil {
				params, _ := tc.Parameters().(map[string]any)
				if params == nil {
					params = make(map[string]any)
				}
				s.onToolCall(tc.Name(), params)
			}

			result, err := llm.ExecuteToolCall(ctx, tc, s.tools...)
			if err != nil {
				if s.onError != nil {
					s.onError(err)
				}
				return errors.WithStack(err)
			}

			if s.onToolResult != nil {
				s.onToolResult(tc.Name(), result.Content())
			}

			// Add tool result to messages
			s.messages = append(s.messages, result)
		}

		// Continue with tool execution loop using non-streaming
		return s.sendMessageWithToolsLoop(ctx)
	}

	// Add assistant message if we have content
	if assistantContent.Len() > 0 {
		s.messages = append(s.messages, llm.NewMessage(llm.RoleAssistant, assistantContent.String()))
	}

	if s.onComplete != nil {
		s.onComplete()
	}

	return nil
}

// sendMessageWithToolsLoop handles the tool execution loop using non-streaming API
func (s *ChatSession) sendMessageWithToolsLoop(ctx context.Context) error {
	// Build options
	completionOpts := []llm.ChatCompletionOptionFunc{
		llm.WithMessages(s.getMessages()...),
		llm.WithTemperature(s.temperature),
	}

	if len(s.tools) > 0 {
		completionOpts = append(completionOpts, llm.WithTools(s.tools...), llm.WithToolChoice(llm.ToolChoiceAuto))
	}

	if s.reasoning != nil {
		completionOpts = append(completionOpts, llm.WithReasoning(s.reasoning))
	}

	// Use non-streaming API for tool execution loop
	res, err := s.client.ChatCompletion(ctx, completionOpts...)
	if err != nil {
		if s.onError != nil {
			s.onError(err)
		}
		return errors.WithStack(err)
	}

	// Handle content response
	if content := res.Message().Content(); content != "" {
		if s.onStreamChunk != nil {
			s.onStreamChunk(content)
		}
	}

	// Handle tool calls
	if len(res.ToolCalls()) > 0 {
		// Ensure all tool calls have valid IDs
		toolCalls := res.ToolCalls()
		for i, tc := range toolCalls {
			if tc.ID() == "" {
				// Create a wrapper with generated ID
				toolCalls[i] = &toolCallWithID{
					ToolCall: tc,
					id:       "tc_" + uuid.New().String()[:8],
				}
			}
		}

		// Add assistant message with tool calls, preserving reasoning if present.
		if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok {
			if r := rr.Reasoning(); r != "" || len(rr.ReasoningDetails()) > 0 {
				s.messages = append(s.messages, llm.NewReasoningToolCallsMessage(r, rr.ReasoningDetails(), toolCalls...))
			} else {
				s.messages = append(s.messages, llm.NewToolCallsMessage(toolCalls...))
			}
		} else {
			s.messages = append(s.messages, llm.NewToolCallsMessage(toolCalls...))
		}

		// Execute tool calls
		for _, tc := range toolCalls {
			if s.onToolCall != nil {
				params, _ := tc.Parameters().(map[string]any)
				if params == nil {
					params = make(map[string]any)
				}
				s.onToolCall(tc.Name(), params)
			}

			result, err := llm.ExecuteToolCall(ctx, tc, s.tools...)
			if err != nil {
				if s.onError != nil {
					s.onError(err)
				}
				return errors.WithStack(err)
			}

			if s.onToolResult != nil {
				s.onToolResult(tc.Name(), result.Content())
			}

			// Add tool result to messages
			s.messages = append(s.messages, result)
		}

		// Continue with tool execution loop
		return s.sendMessageWithToolsLoop(ctx)
	}

	// Add assistant message if we have content
	if res.Message().Content() != "" {
		s.messages = append(s.messages, res.Message())
	}

	if s.onComplete != nil {
		s.onComplete()
	}

	return nil
}

// streamingToolCall accumulates tool call data during streaming
type streamingToolCall struct {
	index      int
	id         string
	name       string
	parameters string
}

// ID implements llm.ToolCall
func (t *streamingToolCall) ID() string {
	return t.id
}

// Name implements llm.ToolCall
func (t *streamingToolCall) Name() string {
	return t.name
}

// Parameters implements llm.ToolCall
func (t *streamingToolCall) Parameters() any {
	return t.parameters
}

// Content implements llm.ToolCall (returns empty for tool calls)
func (t *streamingToolCall) Content() string {
	return ""
}

// Role implements llm.ToolCall
func (t *streamingToolCall) Role() llm.Role {
	return llm.RoleToolCalls
}

// Attachments implements llm.ToolCall
func (t *streamingToolCall) Attachments() []llm.Attachment {
	return nil
}

// ToolCalls implements llm.ToolCallsMessage
func (t *streamingToolCall) ToolCalls() []llm.ToolCall {
	return []llm.ToolCall{t}
}

var _ llm.ToolCall = &streamingToolCall{}

// toolCallWithID wraps a ToolCall to provide a generated ID
type toolCallWithID struct {
	llm.ToolCall
	id string
}

func (t *toolCallWithID) ID() string {
	return t.id
}

var _ llm.ToolCall = &toolCallWithID{}

// getMessages returns all messages including system prompt
func (s *ChatSession) getMessages() []llm.Message {
	messages := make([]llm.Message, 0)

	if s.systemPrompt != "" {
		messages = append(messages, llm.NewMessage(llm.RoleSystem, s.systemPrompt))
	}

	messages = append(messages, s.messages...)

	return messages
}

// GetMessages returns the conversation history
func (s *ChatSession) GetMessages() []llm.Message {
	return s.messages
}

// Clear clears the conversation history
func (s *ChatSession) Clear() {
	s.messages = make([]llm.Message, 0)
}

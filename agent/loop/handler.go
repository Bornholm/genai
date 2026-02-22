package loop

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/agent/todo"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// Handler implements the tool-calling agentic loop
type Handler struct {
	options *Options
}

// Handle implements agent.Handler
func (h *Handler) Handle(ctx context.Context, input agent.Input, emit agent.EmitFunc) error {
	// 1. Build the initial message list
	messages := h.buildInitialMessages(input)

	// Create todo list for this session
	todoList := &todo.List{}

	// Get todo tools
	todoTools := todo.NewTools(todoList, emit)

	// Merge user tools with todo tools
	allTools := append(h.options.Tools, todoTools...)

	// Create context manager
	contextManager := NewContextManager(h.options.MaxTokens, h.options.TokenEstimator, h.options.TruncationStrategy)

	// 2. Enter the loop
	for iteration := 0; iteration < h.options.MaxIterations; iteration++ {
		// 2a. Check ctx.Err()
		if err := ctx.Err(); err != nil {
			return err
		}

		// Apply context window management
		messages = contextManager.Manage(messages)

		// 2b. Call the LLM
		slog.DebugContext(ctx, "calling LLM", slog.Int("iteration", iteration))

		res, err := h.options.Client.ChatCompletion(ctx,
			llm.WithMessages(messages...),
			llm.WithTools(allTools...),
			llm.WithToolChoice(llm.ToolChoiceAuto),
		)
		if err != nil {
			// 2c. Return the error (retry is delegated to llm client middleware)
			return errors.WithStack(err)
		}

		// 2d. Append the assistant message to the history
		// Only append if there's content and no tool calls
		if res.Message() != nil && res.Message().Content() != "" && len(res.ToolCalls()) == 0 {
			messages = append(messages, res.Message())
		}

		// 2e. Extract tool calls from the response
		toolCalls := res.ToolCalls()

		// 2f. If there are NO tool calls
		if len(toolCalls) == 0 {
			// Emit a Complete event with the assistant message content
			if err := emit(agent.NewEvent(agent.EventTypeComplete, &agent.CompleteData{
				Message: res.Message().Content(),
			})); err != nil {
				return errors.WithStack(err)
			}
			// Return nil. The loop is done.
			return nil
		}

		// 2g. If there ARE tool calls
		// Create a tool calls message
		messages = append(messages, llm.NewToolCallsMessage(toolCalls...))

		for _, tc := range toolCalls {
			// Emit a ToolCallStart event
			if err := emit(agent.NewEvent(agent.EventTypeToolCallStart, &agent.ToolCallStartData{
				ID:         tc.ID(),
				Name:       tc.Name(),
				Parameters: tc.Parameters(),
			})); err != nil {
				return errors.WithStack(err)
			}

			var result string

			// Check if approval is required
			if h.requiresApproval(tc.Name()) {
				if h.options.ApprovalFunc != nil {
					argsJSON, _ := json.Marshal(tc.Parameters())
					approved, err := h.options.ApprovalFunc(ctx, tc.Name(), string(argsJSON))
					if err != nil {
						// Return the error - this is a real system failure
						return errors.WithStack(err)
					}
					if !approved {
						result = "The user denied this tool call. Adjust your approach or ask the user for guidance."
					}
				}
			}

			// If not denied, execute the tool
			if result == "" {
				var execErr error
				result, execErr = executeTool(ctx, allTools, tc)
				if execErr != nil {
					// Tool errors are converted to result messages
					result = execErr.Error()
				}
			}

			// Create tool result and append to messages
			toolResult := llm.NewToolResult(result)
			toolMessage := llm.NewToolMessage(tc.ID(), toolResult)
			messages = append(messages, toolMessage)

			// Emit a ToolCallDone event
			if err := emit(agent.NewEvent(agent.EventTypeToolCallDone, &agent.ToolCallDoneData{
				ID:     tc.ID(),
				Name:   tc.Name(),
				Result: result,
			})); err != nil {
				return errors.WithStack(err)
			}
		}

		// 2h. Continue the loop
	}

	// 3. If the loop exits because maxIterations was reached
	if err := emit(agent.NewEvent(agent.EventTypeError, &agent.ErrorData{
		Message: fmt.Sprintf("Exceeded maximum iterations (%d)", h.options.MaxIterations),
	})); err != nil {
		return errors.WithStack(err)
	}

	return ErrIterationBudgetExceeded
}

// requiresApproval checks if a tool requires approval
func (h *Handler) requiresApproval(toolName string) bool {
	if h.options.ApprovalRequiredAll {
		return true
	}
	return h.options.ApprovalRequired[toolName]
}

// buildInitialMessages creates the initial message list
func (h *Handler) buildInitialMessages(input agent.Input) []llm.Message {
	messages := make([]llm.Message, 0, 2)

	// Add system message
	if h.options.SystemPrompt != "" {
		messages = append(messages, llm.NewMessage(llm.RoleSystem, h.options.SystemPrompt))
	}

	// Add user message with attachments
	if len(input.Attachments) > 0 {
		messages = append(messages, llm.NewMessageWithAttachments(llm.RoleUser, input.Message, input.Attachments...))
	} else {
		messages = append(messages, llm.NewMessage(llm.RoleUser, input.Message))
	}

	return messages
}

// NewHandler creates a new loop handler
func NewHandler(funcs ...OptionFunc) (*Handler, error) {
	opts := NewOptions(funcs...)

	if opts.Client == nil {
		return nil, errors.New("client is required")
	}

	if opts.SystemPrompt == "" {
		return nil, errors.New("system prompt is required")
	}

	return &Handler{
		options: opts,
	}, nil
}

// MustNewHandler creates a new loop handler and panics on error
func MustNewHandler(funcs ...OptionFunc) *Handler {
	h, err := NewHandler(funcs...)
	if err != nil {
		panic(err)
	}
	return h
}

var _ agent.Handler = &Handler{}

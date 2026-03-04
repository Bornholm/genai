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

	// 1b. Forced planning step: expose only TodoWrite with tool_choice=required so
	// the model MUST write a structured plan before taking any action.
	// This is skipped when ForcePlanningStep is false or when no tools exist.
	if h.options.ForcePlanningStep && len(allTools) > 0 {
		if err := h.runPlanningStep(ctx, allTools, &messages, emit); err != nil {
			return errors.WithStack(err)
		}
	}

	// 2. Enter the loop
	for iteration := 0; iteration < h.options.MaxIterations; iteration++ {
		// 2a. Check ctx.Err()
		if err := ctx.Err(); err != nil {
			return err
		}

		// Apply context window management
		messages = contextManager.Manage(messages)

		// 2b. Build iteration budget message
		remainingIterations := h.options.MaxIterations - iteration
		budgetMessage := llm.NewMessage(llm.RoleSystem, fmt.Sprintf(
			"You have %d iteration%s remaining to complete your task.",
			remainingIterations,
			map[bool]string{true: "", false: "s"}[remainingIterations == 1],
		))

		// 2c. Call the LLM with the budget message prepended
		slog.DebugContext(ctx, "calling LLM", slog.Int("iteration", iteration), slog.Int("remaining", remainingIterations))

		completionOpts := []llm.ChatCompletionOptionFunc{
			llm.WithMessages(budgetMessage),
			llm.WithMessages(messages...),
			llm.WithTools(allTools...),
			llm.WithToolChoice(llm.ToolChoiceAuto),
		}
		if h.options.Reasoning != nil {
			completionOpts = append(completionOpts, llm.WithReasoning(h.options.Reasoning))
		}

		res, err := h.options.Client.ChatCompletion(ctx, completionOpts...)
		if err != nil {
			// 2d. Return the error (retry is delegated to llm client middleware)
			return errors.WithStack(err)
		}

		// 2e. Emit a reasoning event if the response carries reasoning tokens.
		if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok {
			if reasoning := rr.Reasoning(); reasoning != "" || len(rr.ReasoningDetails()) > 0 {
				agentDetails := make([]agent.ReasoningDetail, 0, len(rr.ReasoningDetails()))
				for _, d := range rr.ReasoningDetails() {
					agentDetails = append(agentDetails, agent.ReasoningDetail{
						ID:        d.ID,
						Type:      string(d.Type),
						Text:      d.Text,
						Summary:   d.Summary,
						Data:      d.Data,
						Format:    d.Format,
						Index:     d.Index,
						Signature: d.Signature,
					})
				}
				if err := emit(agent.NewEvent(agent.EventTypeReasoning, &agent.ReasoningData{
					Reasoning:        reasoning,
					ReasoningDetails: agentDetails,
				})); err != nil {
					return errors.WithStack(err)
				}
				slog.DebugContext(ctx, "reasoning tokens received",
					slog.Int("iteration", iteration),
					slog.Int("details", len(agentDetails)),
				)
			}
		}

		// 2f. Append the assistant message to the history
		// Only append if there's content and no tool calls
		if res.Message() != nil && res.Message().Content() != "" && len(res.ToolCalls()) == 0 {
			messages = append(messages, res.Message())
		}

		// 2g. Extract tool calls from the response
		toolCalls := res.ToolCalls()

		// 2h. If there are NO tool calls
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

		// 2i. If there ARE tool calls
		// Create a tool calls message, preserving reasoning if the response carries it.
		// Reasoning models (e.g. Claude, GPT-5) require reasoning blocks to be passed
		// back alongside tool calls so the model can continue its chain-of-thought.
		if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok {
			reasoning := rr.Reasoning()
			reasoningDetails := rr.ReasoningDetails()
			if reasoning != "" || len(reasoningDetails) > 0 {
				messages = append(messages, llm.NewReasoningToolCallsMessage(reasoning, reasoningDetails, toolCalls...))
			} else {
				messages = append(messages, llm.NewToolCallsMessage(toolCalls...))
			}
		} else {
			messages = append(messages, llm.NewToolCallsMessage(toolCalls...))
		}

		// 2j. Execute each tool call
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

			// Truncate tool result if too large to prevent context overflow
			result = h.truncateToolResult(result)

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

		// 2k. Continue the loop
	}

	// 3. If the loop exits because maxIterations was reached, generate a summary
	return h.generateBudgetExceededSummary(ctx, messages, emit)
}

// generateBudgetExceededSummary makes a final LLM call to summarize what was accomplished
// within the iteration budget when the agent exceeds its maximum iterations.
func (h *Handler) generateBudgetExceededSummary(ctx context.Context, messages []llm.Message, emit agent.EmitFunc) error {
	slog.DebugContext(ctx, "generating budget exceeded summary")

	summaryPrompt := `You have exceeded your iteration budget and cannot continue using tools.
Please provide a summary of what you accomplished during this session and what remains unfinished.
Be specific about progress made, any obstacles encountered, and recommendations for completing the remaining work.`

	summaryMessage := llm.NewMessage(llm.RoleUser, summaryPrompt)
	summaryMessages := append(messages, summaryMessage)

	res, err := h.options.Client.ChatCompletion(ctx,
		llm.WithMessages(summaryMessages...),
		llm.WithTools(), // No tools for summary
	)
	if err != nil {
		// If summary call fails, emit a basic error event
		if emitErr := emit(agent.NewEvent(agent.EventTypeError, &agent.ErrorData{
			Message: fmt.Sprintf("Iteration budget exceeded (%d). Failed to generate summary: %v", h.options.MaxIterations, err),
		})); emitErr != nil {
			return errors.WithStack(emitErr)
		}
		return ErrIterationBudgetExceeded
	}

	// Emit the summary as a Complete event
	summary := res.Message().Content()
	if summary == "" {
		summary = fmt.Sprintf("Iteration budget exceeded (%d). Unable to generate summary.", h.options.MaxIterations)
	}

	if err := emit(agent.NewEvent(agent.EventTypeComplete, &agent.CompleteData{
		Message: summary,
	})); err != nil {
		return errors.WithStack(err)
	}

	return nil
}

// runPlanningStep makes one dedicated LLM call with only the TodoWrite tool exposed
// and tool_choice=required, guaranteeing the model writes a plan before acting.
func (h *Handler) runPlanningStep(ctx context.Context, allTools []llm.Tool, messages *[]llm.Message, emit agent.EmitFunc) error {
	// Find the TodoWrite tool that was injected by the loop
	var todoWriteTool llm.Tool
	for _, t := range allTools {
		if t.Name() == "TodoWrite" {
			todoWriteTool = t
			break
		}
	}
	if todoWriteTool == nil {
		// TodoWrite not present — nothing to force
		return nil
	}

	slog.DebugContext(ctx, "running forced planning step")

	planOpts := []llm.ChatCompletionOptionFunc{
		llm.WithMessages(*messages...),
		llm.WithTools(todoWriteTool),
		llm.WithToolChoice(llm.ToolChoiceRequired),
	}
	if h.options.Reasoning != nil {
		planOpts = append(planOpts, llm.WithReasoning(h.options.Reasoning))
	}

	res, err := h.options.Client.ChatCompletion(ctx, planOpts...)
	if err != nil {
		return errors.WithStack(err)
	}

	// Emit reasoning event if present
	if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok {
		if reasoning := rr.Reasoning(); reasoning != "" || len(rr.ReasoningDetails()) > 0 {
			agentDetails := make([]agent.ReasoningDetail, 0, len(rr.ReasoningDetails()))
			for _, d := range rr.ReasoningDetails() {
				agentDetails = append(agentDetails, agent.ReasoningDetail{
					ID:        d.ID,
					Type:      string(d.Type),
					Text:      d.Text,
					Summary:   d.Summary,
					Data:      d.Data,
					Format:    d.Format,
					Index:     d.Index,
					Signature: d.Signature,
				})
			}
			if err := emit(agent.NewEvent(agent.EventTypeReasoning, &agent.ReasoningData{
				Reasoning:        reasoning,
				ReasoningDetails: agentDetails,
			})); err != nil {
				return errors.WithStack(err)
			}
		}
	}

	// Execute the TodoWrite tool call(s) returned by the planning step
	toolCalls := res.ToolCalls()
	if len(toolCalls) == 0 {
		// Model produced no tool call despite required — skip quietly
		return nil
	}

	// Build the tool calls message, preserving reasoning for models that need it
	var tcMsg llm.Message
	if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok {
		if r := rr.Reasoning(); r != "" || len(rr.ReasoningDetails()) > 0 {
			tcMsg = llm.NewReasoningToolCallsMessage(r, rr.ReasoningDetails(), toolCalls...)
		}
	}
	if tcMsg == nil {
		tcMsg = llm.NewToolCallsMessage(toolCalls...)
	}
	*messages = append(*messages, tcMsg)

	for _, tc := range toolCalls {
		if err := emit(agent.NewEvent(agent.EventTypeToolCallStart, &agent.ToolCallStartData{
			ID:         tc.ID(),
			Name:       tc.Name(),
			Parameters: tc.Parameters(),
		})); err != nil {
			return errors.WithStack(err)
		}

		result, execErr := executeTool(ctx, allTools, tc)
		if execErr != nil {
			result = execErr.Error()
		}

		toolResult := llm.NewToolResult(result)
		toolMessage := llm.NewToolMessage(tc.ID(), toolResult)
		*messages = append(*messages, toolMessage)

		if err := emit(agent.NewEvent(agent.EventTypeToolCallDone, &agent.ToolCallDoneData{
			ID:     tc.ID(),
			Name:   tc.Name(),
			Result: result,
		})); err != nil {
			return errors.WithStack(err)
		}
	}

	return nil
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

// truncateToolResult truncates the tool result if it exceeds the max tool result tokens
func (h *Handler) truncateToolResult(result string) string {
	maxTokens := h.options.MaxToolResultTokens
	if maxTokens <= 0 {
		return result
	}

	// Estimate tokens using the token estimator
	estimatedTokens := h.options.TokenEstimator(result)
	if estimatedTokens <= maxTokens {
		return result
	}

	// Calculate the max characters based on the token estimator ratio
	// If we estimate 1 token per 4 chars, then maxChars = maxTokens * 4
	maxChars := maxTokens * 4

	// Ensure we have enough room for the truncation notice
	truncationNotice := "\n\n[Output truncated due to size. Showing first %d characters.]"
	noticeLen := len(fmt.Sprintf(truncationNotice, maxChars))
	if maxChars > noticeLen+10 {
		maxChars -= noticeLen
	} else {
		maxChars = 1000 // Minimum
	}

	if len(result) > maxChars {
		result = result[:maxChars] + fmt.Sprintf(truncationNotice, maxChars)
	}

	return result
}

var _ agent.Handler = &Handler{}

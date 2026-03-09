package loop

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"

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

		// 2c. Call the LLM (streaming when supported, non-streaming otherwise)
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

		result, err := h.doLLMCall(ctx, completionOpts, emit)
		if err != nil {
			return errors.WithStack(err)
		}

		// 2d. Append the assistant message to the history only when there are no tool calls.
		if result.content != "" && len(result.toolCalls) == 0 {
			messages = append(messages, llm.NewMessage(llm.RoleAssistant, result.content))
		}

		// 2e. If there are NO tool calls, emit Complete and finish.
		if len(result.toolCalls) == 0 {
			if err := emit(agent.NewEvent(agent.EventTypeComplete, &agent.CompleteData{
				Message: result.content,
			})); err != nil {
				return errors.WithStack(err)
			}
			return nil
		}

		// 2f. Append the tool calls message (preserving reasoning blocks for models that need them).
		messages = append(messages, h.makeToolCallsMessage(result))

		// 2g. Emit ToolCallStart for every tool call (sequential to preserve ordering).
		for _, tc := range result.toolCalls {
			if err := emit(agent.NewEvent(agent.EventTypeToolCallStart, &agent.ToolCallStartData{
				ID:         tc.ID(),
				Name:       tc.Name(),
				Parameters: tc.Parameters(),
			})); err != nil {
				return errors.WithStack(err)
			}
		}

		// 2h. Check approval sequentially (may be interactive/blocking).
		toolResults := make([]string, len(result.toolCalls))
		for i, tc := range result.toolCalls {
			if h.requiresApproval(tc.Name()) && h.options.ApprovalFunc != nil {
				argsJSON, _ := json.Marshal(tc.Parameters())
				approved, approvalErr := h.options.ApprovalFunc(ctx, tc.Name(), string(argsJSON))
				if approvalErr != nil {
					return errors.WithStack(approvalErr)
				}
				if !approved {
					toolResults[i] = "The user denied this tool call. Adjust your approach or ask the user for guidance."
				}
			}
		}

		// 2i. Execute approved tools in parallel.
		var wg sync.WaitGroup
		for i, tc := range result.toolCalls {
			if toolResults[i] != "" {
				continue // already set (denied)
			}
			wg.Add(1)
			go func(i int, tc llm.ToolCall) {
				defer wg.Done()
				r, execErr := executeTool(ctx, allTools, tc)
				if execErr != nil {
					r = execErr.Error()
				}
				toolResults[i] = h.truncateToolResult(r)
			}(i, tc)
		}
		wg.Wait()

		// 2j. Append tool results and emit ToolCallDone events in order.
		for i, tc := range result.toolCalls {
			messages = append(messages, llm.NewToolMessage(tc.ID(), llm.NewToolResult(toolResults[i])))
			if err := emit(agent.NewEvent(agent.EventTypeToolCallDone, &agent.ToolCallDoneData{
				ID:     tc.ID(),
				Name:   tc.Name(),
				Result: toolResults[i],
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
	var todoWriteTool llm.Tool
	for _, t := range allTools {
		if t.Name() == "TodoWrite" {
			todoWriteTool = t
			break
		}
	}
	if todoWriteTool == nil {
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

	result, err := h.doLLMCall(ctx, planOpts, emit)
	if err != nil {
		return errors.WithStack(err)
	}

	if len(result.toolCalls) == 0 {
		return nil
	}

	*messages = append(*messages, h.makeToolCallsMessage(result))

	for _, tc := range result.toolCalls {
		if err := emit(agent.NewEvent(agent.EventTypeToolCallStart, &agent.ToolCallStartData{
			ID:         tc.ID(),
			Name:       tc.Name(),
			Parameters: tc.Parameters(),
		})); err != nil {
			return errors.WithStack(err)
		}

		r, execErr := executeTool(ctx, allTools, tc)
		if execErr != nil {
			r = execErr.Error()
		}

		*messages = append(*messages, llm.NewToolMessage(tc.ID(), llm.NewToolResult(r)))

		if err := emit(agent.NewEvent(agent.EventTypeToolCallDone, &agent.ToolCallDoneData{
			ID:     tc.ID(),
			Name:   tc.Name(),
			Result: r,
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

// llmTurnResult holds the accumulated result of one LLM call, whether streamed or not.
type llmTurnResult struct {
	content          string
	toolCalls        []llm.ToolCall
	reasoning        string
	reasoningDetails []llm.ReasoningDetail
}

// streamToolCallAcc accumulates incremental tool call data from streaming chunks.
type streamToolCallAcc struct {
	id     string
	name   string
	params strings.Builder
}

// doLLMCall performs one LLM call, using streaming when the client supports it.
// In streaming mode, text deltas are emitted as EventTypeTextDelta events.
// Reasoning events are emitted in both modes via emitReasoningIfPresent.
func (h *Handler) doLLMCall(ctx context.Context, completionOpts []llm.ChatCompletionOptionFunc, emit agent.EmitFunc) (*llmTurnResult, error) {
	if sc, ok := h.options.Client.(llm.ChatCompletionStreamingClient); ok {
		return h.doStreamingLLMCall(ctx, sc, completionOpts, emit)
	}
	return h.doNonStreamingLLMCall(ctx, completionOpts, emit)
}

// doNonStreamingLLMCall calls ChatCompletion and extracts the result.
func (h *Handler) doNonStreamingLLMCall(ctx context.Context, completionOpts []llm.ChatCompletionOptionFunc, emit agent.EmitFunc) (*llmTurnResult, error) {
	res, err := h.options.Client.ChatCompletion(ctx, completionOpts...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	result := &llmTurnResult{
		toolCalls: res.ToolCalls(),
	}
	if res.Message() != nil {
		result.content = res.Message().Content()
	}
	if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok {
		result.reasoning = rr.Reasoning()
		result.reasoningDetails = rr.ReasoningDetails()
	}

	if err := h.emitReasoningIfPresent(result, emit); err != nil {
		return nil, errors.WithStack(err)
	}
	return result, nil
}

// doStreamingLLMCall streams a response, emitting EventTypeTextDelta for each text chunk.
// Tool call deltas are accumulated and reconstructed in index order.
func (h *Handler) doStreamingLLMCall(ctx context.Context, sc llm.ChatCompletionStreamingClient, completionOpts []llm.ChatCompletionOptionFunc, emit agent.EmitFunc) (*llmTurnResult, error) {
	ch, err := sc.ChatCompletionStream(ctx, completionOpts...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	var (
		contentBuf    strings.Builder
		reasoningBuf  strings.Builder
		reasoningDets []llm.ReasoningDetail
		toolCallAccs  = make(map[int]*streamToolCallAcc)
		// Once a tool call delta is received, stop emitting text deltas to avoid
		// displaying raw JSON arguments that some providers stream via content.
		seenToolCallDelta bool
	)

	for chunk := range ch {
		if chunk.Error() != nil {
			return nil, errors.WithStack(chunk.Error())
		}
		if chunk.IsComplete() {
			break
		}
		delta := chunk.Delta()
		if delta == nil {
			continue
		}

		// Tool call deltas — accumulate by index (processed before text so we
		// can stop text emission as soon as a tool call delta is seen).
		for _, tc := range delta.ToolCalls() {
			seenToolCallDelta = true
			idx := tc.Index()
			if _, exists := toolCallAccs[idx]; !exists {
				toolCallAccs[idx] = &streamToolCallAcc{}
			}
			acc := toolCallAccs[idx]
			if tc.ID() != "" {
				acc.id = tc.ID()
			}
			if tc.Name() != "" {
				acc.name = tc.Name()
			}
			acc.params.WriteString(tc.ParametersDelta())
		}

		// Text content → emit delta event only when no tool call deltas have
		// been seen yet (prevents streaming raw JSON tool-call arguments).
		if text := delta.Content(); text != "" {
			contentBuf.WriteString(text)
			if !seenToolCallDelta {
				if err := emit(agent.NewEvent(agent.EventTypeTextDelta, &agent.TextDeltaData{Delta: text})); err != nil {
					return nil, errors.WithStack(err)
				}
			}
		}

		// Reasoning tokens
		if rsd, ok := delta.(llm.ReasoningStreamDelta); ok {
			if r := rsd.Reasoning(); r != "" {
				reasoningBuf.WriteString(r)
			}
			reasoningDets = append(reasoningDets, rsd.ReasoningDetails()...)
		}
	}

	// Reconstruct tool calls in index order.
	indices := make([]int, 0, len(toolCallAccs))
	for idx := range toolCallAccs {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	toolCalls := make([]llm.ToolCall, 0, len(indices))
	for _, idx := range indices {
		acc := toolCallAccs[idx]
		toolCalls = append(toolCalls, llm.NewToolCall(acc.id, acc.name, acc.params.String()))
	}

	result := &llmTurnResult{
		content:          contentBuf.String(),
		toolCalls:        toolCalls,
		reasoning:        reasoningBuf.String(),
		reasoningDetails: reasoningDets,
	}

	if err := h.emitReasoningIfPresent(result, emit); err != nil {
		return nil, errors.WithStack(err)
	}
	return result, nil
}

// emitReasoningIfPresent emits an EventTypeReasoning event when the result carries reasoning tokens.
func (h *Handler) emitReasoningIfPresent(result *llmTurnResult, emit agent.EmitFunc) error {
	if result.reasoning == "" && len(result.reasoningDetails) == 0 {
		return nil
	}
	agentDetails := make([]agent.ReasoningDetail, 0, len(result.reasoningDetails))
	for _, d := range result.reasoningDetails {
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
	return emit(agent.NewEvent(agent.EventTypeReasoning, &agent.ReasoningData{
		Reasoning:        result.reasoning,
		ReasoningDetails: agentDetails,
	}))
}

// makeToolCallsMessage builds the assistant tool calls message, preserving reasoning
// blocks for models (e.g. Claude) that require them across turns.
func (h *Handler) makeToolCallsMessage(result *llmTurnResult) llm.Message {
	if result.reasoning != "" || len(result.reasoningDetails) > 0 {
		return llm.NewReasoningToolCallsMessage(result.reasoning, result.reasoningDetails, result.toolCalls...)
	}
	return llm.NewToolCallsMessage(result.toolCalls...)
}

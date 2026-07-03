package loop

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/llm"
)

// MockChatCompletionClient implements llm.ChatCompletionClient for testing
type MockChatCompletionClient struct {
	responses []MockResponse
	callCount int
	// capturedMessages records the messages passed to each ChatCompletion call,
	// letting tests assert on what the agent actually sent (e.g. the budget message).
	capturedMessages [][]llm.Message
}

type MockResponse struct {
	Message   llm.Message
	ToolCalls []llm.ToolCall
	Err       error
}

func (m *MockChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	m.capturedMessages = append(m.capturedMessages, llm.NewChatCompletionOptions(funcs...).Messages)
	if m.callCount >= len(m.responses) {
		return nil, errors.New("no more mock responses")
	}
	resp := m.responses[m.callCount]
	m.callCount++
	return &MockChatCompletionResponse{
		message:   resp.Message,
		toolCalls: resp.ToolCalls,
	}, resp.Err
}

type MockChatCompletionResponse struct {
	message   llm.Message
	toolCalls []llm.ToolCall
}

func (r *MockChatCompletionResponse) Message() llm.Message {
	return r.message
}

func (r *MockChatCompletionResponse) ToolCalls() []llm.ToolCall {
	return r.toolCalls
}

func (r *MockChatCompletionResponse) Usage() llm.ChatCompletionUsage {
	return nil
}

// MockToolCall implements llm.ToolCall for testing
type MockToolCall struct {
	id         string
	name       string
	parameters any
}

func (m *MockToolCall) ID() string                    { return m.id }
func (m *MockToolCall) Name() string                  { return m.name }
func (m *MockToolCall) Parameters() any               { return m.parameters }
func (m *MockToolCall) Role() llm.Role                { return llm.RoleToolCalls }
func (m *MockToolCall) Content() string               { return "" }
func (m *MockToolCall) Attachments() []llm.Attachment { return nil }
func (m *MockToolCall) ToolCalls() []llm.ToolCall     { return []llm.ToolCall{m} }

// MockTool implements llm.Tool for testing
type MockTool struct {
	name        string
	description string
	execute     func(ctx context.Context, params map[string]any) (llm.ToolResult, error)
}

func (m *MockTool) Name() string               { return m.name }
func (m *MockTool) Description() string        { return m.description }
func (m *MockTool) Parameters() map[string]any { return nil }
func (m *MockTool) Execute(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
	return m.execute(ctx, params)
}

func TestHandler_SimpleCompletion(t *testing.T) {
	// Test: LLM returns text with no tool calls → handler emits Complete, returns nil
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				Message: llm.NewMessage(llm.RoleAssistant, "Hello, I can help you with that."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Hi"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	if len(events) != 1 {
		t.Errorf("expected 1 event, got %d", len(events))
	}

	if events[0].Type() != agent.EventTypeComplete {
		t.Errorf("expected EventTypeComplete, got %s", events[0].Type())
	}

	data := events[0].Data().(*agent.CompleteData)
	if data.Message != "Hello, I can help you with that." {
		t.Errorf("expected message 'Hello, I can help you with that.', got '%s'", data.Message)
	}
}

func TestHandler_SingleToolCall(t *testing.T) {
	// Test: LLM returns one tool call → tool executes → LLM returns text → Complete
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{id: "1", name: "test_tool", parameters: map[string]any{"arg": "value"}}},
			},
			{
				Message: llm.NewMessage(llm.RoleAssistant, "I used the tool."),
			},
		},
	}

	tool := &MockTool{
		name:        "test_tool",
		description: "A test tool",
		execute: func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			return llm.NewToolResult("tool result"), nil
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
		WithTools(tool),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Use the tool"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	// Should have: ToolCallStart, ToolCallDone, Complete
	if len(events) != 3 {
		t.Errorf("expected 3 events, got %d", len(events))
	}

	if events[0].Type() != agent.EventTypeToolCallStart {
		t.Errorf("expected EventTypeToolCallStart, got %s", events[0].Type())
	}

	if events[1].Type() != agent.EventTypeToolCallDone {
		t.Errorf("expected EventTypeToolCallDone, got %s", events[1].Type())
	}

	if events[2].Type() != agent.EventTypeComplete {
		t.Errorf("expected EventTypeComplete, got %s", events[2].Type())
	}
}

func TestHandler_ToolErrorRecovery(t *testing.T) {
	// Test: LLM calls a tool → tool returns error → error becomes tool result → LLM adjusts and completes
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{id: "1", name: "failing_tool", parameters: map[string]any{}}},
			},
			{
				Message: llm.NewMessage(llm.RoleAssistant, "I handled the error."),
			},
		},
	}

	tool := &MockTool{
		name:        "failing_tool",
		description: "A tool that fails",
		execute: func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			return nil, errors.New("tool failed")
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
		WithTools(tool),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Test"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	// Tool error should be converted to result, loop should continue
	if events[1].Type() != agent.EventTypeToolCallDone {
		t.Errorf("expected EventTypeToolCallDone, got %s", events[1].Type())
	}

	data := events[1].Data().(*agent.ToolCallDoneData)
	if data.Result == "" {
		t.Error("expected non-empty result from tool error")
	}
}

func TestHandler_UnknownTool(t *testing.T) {
	// Test: LLM calls a tool name that doesn't exist → result is "Unknown tool" message → LLM adjusts
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{id: "1", name: "unknown_tool", parameters: map[string]any{}}},
			},
			{
				Message: llm.NewMessage(llm.RoleAssistant, "I see that tool doesn't exist."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
		WithTools(), // No tools provided
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Test"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	data := events[1].Data().(*agent.ToolCallDoneData)
	if data.Result == "" {
		t.Error("expected non-empty result for unknown tool")
	}
}

func TestHandler_ContextCancellation(t *testing.T) {
	// Test: Cancel ctx during the loop → handler returns context error
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{id: "1", name: "test_tool", parameters: map[string]any{}}},
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err = handler.Handle(ctx, agent.NewInput("Test"), func(evt agent.Event) error {
		return nil
	})

	if err == nil {
		t.Error("expected context canceled error")
	}
}

func TestHandler_IterationBudget(t *testing.T) {
	// Test: Set maxIterations to 2, LLM always returns tool calls → handler emits Complete with summary
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{ToolCalls: []llm.ToolCall{&MockToolCall{id: "1", name: "test_tool", parameters: map[string]any{}}}},
			{ToolCalls: []llm.ToolCall{&MockToolCall{id: "2", name: "test_tool", parameters: map[string]any{}}}},
			// Summary call returns text
			{Message: llm.NewMessage(llm.RoleAssistant, "Summary of work done")},
		},
	}

	tool := &MockTool{
		name:        "test_tool",
		description: "A test tool",
		execute: func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			return llm.NewToolResult("result"), nil
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
		WithTools(tool),
		WithMaxIterations(2),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Test"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	// Should return nil (not error) and emit Complete with summary
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	// Should have emitted Complete event with summary
	if len(events) == 0 {
		t.Error("expected at least one event")
	}

	lastEvent := events[len(events)-1]
	if lastEvent.Type() != agent.EventTypeComplete {
		t.Errorf("expected EventTypeComplete, got %s", lastEvent.Type())
	}

	data := lastEvent.Data().(*agent.CompleteData)
	if data.Message == "" {
		t.Error("expected non-empty summary message")
	}
}

func TestHandler_ApprovalDenied(t *testing.T) {
	// Test: Configure approval for a tool, deny it → loop continues with denial message
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{id: "1", name: "dangerous_tool", parameters: map[string]any{}}},
			},
			{
				Message: llm.NewMessage(llm.RoleAssistant, "I understand the tool was denied."),
			},
		},
	}

	tool := &MockTool{
		name:        "dangerous_tool",
		description: "A dangerous tool",
		execute: func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			return llm.NewToolResult("This should not be called"), nil
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
		WithTools(tool),
		WithApprovalRequiredTools("dangerous_tool"),
		WithApprovalFunc(func(ctx interface{ Done() <-chan struct{} }, toolName string, arguments string) (bool, error) {
			return false, nil // Deny
		}),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Test"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	data := events[1].Data().(*agent.ToolCallDoneData)
	if data.Result == "" {
		t.Error("expected denial message in result")
	}
}

func TestHandler_Attachments(t *testing.T) {
	// Test: Input includes attachments → verify they appear in the first user message sent to the LLM
	var receivedMessages []llm.Message
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				Message: llm.NewMessage(llm.RoleAssistant, "I see the attachment."),
			},
		},
	}

	// We need to capture the messages sent to the LLM
	// For this test, we'll just verify the handler doesn't error with attachments
	attachment, err := llm.NewBase64Attachment(llm.AttachmentTypeImage, "image/png", "dGVzdA==")
	if err != nil {
		t.Fatalf("failed to create attachment: %v", err)
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	err = handler.Handle(context.Background(), agent.NewInput("Look at this image", attachment), func(evt agent.Event) error {
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	_ = receivedMessages // Messages were sent (we can't easily verify without modifying the mock)
}

func TestHandler_TodoTools(t *testing.T) {
	// Test: LLM calls TodoWrite then TodoRead then marks done → verify state consistency and events emitted
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{
					id:         "1",
					name:       "TodoWrite",
					parameters: map[string]any{"items": []any{map[string]any{"id": "1", "content": "Task 1", "status": "pending"}}},
				}},
			},
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{
					id:         "2",
					name:       "TodoRead",
					parameters: map[string]any{},
				}},
			},
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{
					id:         "3",
					name:       "TodoWrite",
					parameters: map[string]any{"items": []any{map[string]any{"id": "1", "content": "Task 1", "status": "done"}}},
				}},
			},
			{
				Message: llm.NewMessage(llm.RoleAssistant, "I've updated the todo list."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var todoEvents []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Create a todo list"), func(evt agent.Event) error {
		if evt.Type() == agent.EventTypeTodoUpdated {
			todoEvents = append(todoEvents, evt)
		}
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	if len(todoEvents) == 0 {
		t.Error("expected at least one TodoUpdated event")
	}
}

func TestHandler_PendingTodoBlocksCompletion(t *testing.T) {
	// Test: LLM tries to finish early while a todo item is still pending →
	// continuation prompt injected → LLM marks item done → Complete emitted.
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{
					id:         "1",
					name:       "TodoWrite",
					parameters: map[string]any{"items": []any{map[string]any{"id": "1", "content": "Do the thing", "status": "pending"}}},
				}},
			},
			{
				// Premature finish — todo still has a pending item
				Message: llm.NewMessage(llm.RoleAssistant, "I think I'm done."),
			},
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{
					id:         "2",
					name:       "TodoWrite",
					parameters: map[string]any{"items": []any{map[string]any{"id": "1", "content": "Do the thing", "status": "done"}}},
				}},
			},
			{
				Message: llm.NewMessage(llm.RoleAssistant, "All done now."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Do the thing"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	lastEvent := events[len(events)-1]
	if lastEvent.Type() != agent.EventTypeComplete {
		t.Errorf("expected EventTypeComplete as last event, got %s", lastEvent.Type())
	}

	data := lastEvent.Data().(*agent.CompleteData)
	if data.Message != "All done now." {
		t.Errorf("expected final message 'All done now.', got '%s'", data.Message)
	}

	for _, evt := range events {
		if evt.Type() == agent.EventTypeComplete {
			d := evt.Data().(*agent.CompleteData)
			if d.Message == "I think I'm done." {
				t.Error("premature completion text should not have been emitted as a Complete event")
			}
		}
	}

	if client.callCount != 4 {
		t.Errorf("expected 4 LLM calls, got %d", client.callCount)
	}
}

func TestHandler_EmptyTodoAllowsCompletion(t *testing.T) {
	// Test: todo list never written (stays empty) → LLM finishes immediately → Complete emitted without injection.
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				Message: llm.NewMessage(llm.RoleAssistant, "Done, nothing to track."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Quick question"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	if len(events) != 1 || events[0].Type() != agent.EventTypeComplete {
		t.Errorf("expected single Complete event, got %d events", len(events))
	}

	if client.callCount != 1 {
		t.Errorf("expected 1 LLM call, got %d", client.callCount)
	}
}

func TestHandler_PlanningStepRetryOnInvalidJSON(t *testing.T) {
	// Test: ForcePlanningStep=true, first TodoWrite call has malformed JSON params (string, not map)
	// → planning step retries → second call succeeds → loop completes normally.
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			// Planning attempt 1: TodoWrite with a string param (simulates malformed streaming JSON)
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{
					id:         "p1",
					name:       "TodoWrite",
					parameters: "not valid json {{{",
				}},
			},
			// Planning attempt 2 (retry): TodoWrite with valid params
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{
					id:         "p2",
					name:       "TodoWrite",
					parameters: map[string]any{"items": []any{map[string]any{"id": "1", "content": "Do the thing", "status": "pending"}}},
				}},
			},
			// Main loop: mark done
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{
					id:         "m1",
					name:       "TodoWrite",
					parameters: map[string]any{"items": []any{map[string]any{"id": "1", "content": "Do the thing", "status": "done"}}},
				}},
			},
			// Final response
			{
				Message: llm.NewMessage(llm.RoleAssistant, "All done."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
		WithForcePlanningStep(true),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Do the thing"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	lastEvent := events[len(events)-1]
	if lastEvent.Type() != agent.EventTypeComplete {
		t.Errorf("expected EventTypeComplete, got %s", lastEvent.Type())
	}

	if client.callCount != 4 {
		t.Errorf("expected 4 LLM calls, got %d", client.callCount)
	}
}

func TestHandler_FinalInstructionInjectedOnce(t *testing.T) {
	// Test: WithFinalInstruction set. The LLM first tries to finish with plain text →
	// the final instruction is injected once and the loop runs one more turn → the LLM
	// finishes again → Complete emitted. The injection must be one-shot: if it repeated,
	// the mock would run out of responses and error.
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				Message: llm.NewMessage(llm.RoleAssistant, "I think I'm done."),
			},
			{
				Message: llm.NewMessage(llm.RoleAssistant, "Confirmed, report exported."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
		WithFinalInstruction("Make sure you exported your report before concluding."),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Do the thing"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	// One extra turn: the instruction forced a second LLM call, no more.
	if client.callCount != 2 {
		t.Errorf("expected 2 LLM calls (one-shot injection), got %d", client.callCount)
	}

	if len(events) != 1 || events[0].Type() != agent.EventTypeComplete {
		t.Fatalf("expected single Complete event, got %d events", len(events))
	}

	data := events[0].Data().(*agent.CompleteData)
	if data.Message != "Confirmed, report exported." {
		t.Errorf("expected final message 'Confirmed, report exported.', got '%s'", data.Message)
	}
}

func TestHandler_NoFinalInstructionNoExtraTurn(t *testing.T) {
	// Regression: without a final instruction the agent completes on the first turn.
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				Message: llm.NewMessage(llm.RoleAssistant, "Done."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Quick question"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	if client.callCount != 1 {
		t.Errorf("expected 1 LLM call, got %d", client.callCount)
	}

	if len(events) != 1 || events[0].Type() != agent.EventTypeComplete {
		t.Errorf("expected single Complete event, got %d events", len(events))
	}
}

func TestHandler_FinalInstructionAppliedOnBudgetExceeded(t *testing.T) {
	// Test: the agent spends its entire iteration budget on tool calls (never stops
	// voluntarily), so the in-loop one-shot final instruction is never injected. On
	// budget exhaustion the handler must still give the final instruction a bounded
	// grace window, letting the agent perform a last action (here: send_report) before
	// the budget-exceeded summary is produced.
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			// Budget is 2 iterations, both tool calls → budget exhausted.
			{ToolCalls: []llm.ToolCall{&MockToolCall{id: "1", name: "test_tool", parameters: map[string]any{}}}},
			{ToolCalls: []llm.ToolCall{&MockToolCall{id: "2", name: "test_tool", parameters: map[string]any{}}}},
			// Grace window: the agent honors the final instruction by sending the report...
			{ToolCalls: []llm.ToolCall{&MockToolCall{id: "3", name: "send_report", parameters: map[string]any{}}}},
			// ...then finishes with plain text (no tool call) → grace window ends early.
			{Message: llm.NewMessage(llm.RoleAssistant, "Report exported.")},
			// Budget-exceeded summary call.
			{Message: llm.NewMessage(llm.RoleAssistant, "Summary of work done.")},
		},
	}

	testTool := &MockTool{
		name:        "test_tool",
		description: "A test tool",
		execute: func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			return llm.NewToolResult("result"), nil
		},
	}
	var reportSent bool
	sendReport := &MockTool{
		name:        "send_report",
		description: "Send the final report",
		execute: func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			reportSent = true
			return llm.NewToolResult("report sent"), nil
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
		WithTools(testTool, sendReport),
		WithMaxIterations(2),
		WithFinalInstruction("Make sure you exported your report before concluding."),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Audit the code"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	// 2 loop turns + 2 grace turns + 1 summary.
	if client.callCount != 5 {
		t.Errorf("expected 5 LLM calls, got %d", client.callCount)
	}

	// The final instruction must have driven send_report during the grace window.
	if !reportSent {
		t.Error("expected send_report to be called via the final-instruction grace window")
	}

	// The budget-exceeded condition must be signaled to callers.
	var sawBudgetExceeded bool
	for _, evt := range events {
		if evt.Type() == agent.EventTypeBudgetExceeded {
			sawBudgetExceeded = true
		}
	}
	if !sawBudgetExceeded {
		t.Error("expected an EventTypeBudgetExceeded event")
	}

	// The last event is the summary Complete.
	last := events[len(events)-1]
	if last.Type() != agent.EventTypeComplete {
		t.Fatalf("expected last event EventTypeComplete, got %s", last.Type())
	}
	if data := last.Data().(*agent.CompleteData); data.Message != "Summary of work done." {
		t.Errorf("expected summary message 'Summary of work done.', got '%s'", data.Message)
	}
}

func TestHandler_BudgetMessageWindsDownNearLimit(t *testing.T) {
	// buildBudgetMessage must escalate to an imperative wind-down once few iterations
	// remain, so the agent stops exploring and delivers its result while it still has
	// budget instead of being cut off mid-exploration.
	handler, err := NewHandler(
		WithClient(&MockChatCompletionClient{}),
		WithSystemPrompt("You are a helpful assistant."),
		WithMaxIterations(100),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	// Early iteration: gentle reminder to keep the roadmap aligned.
	early := handler.buildBudgetMessage(0)
	if !strings.Contains(early.Content(), "remaining to complete your task") {
		t.Errorf("expected the gentle reminder early on, got: %q", early.Content())
	}

	// Final stretch (remaining = 5 <= MaxIterations/10 = 10): imperative wind-down.
	late := handler.buildBudgetMessage(95)
	if !strings.Contains(late.Content(), "STOP gathering new information") {
		t.Errorf("expected the imperative wind-down near the limit, got: %q", late.Content())
	}
}

// messagesContainSystem reports whether any system message in the slice contains substr.
func messagesContainSystem(messages []llm.Message, substr string) bool {
	for _, m := range messages {
		if m.Role() == llm.RoleSystem && strings.Contains(m.Content(), substr) {
			return true
		}
	}
	return false
}

func TestHandler_PlanningStepReceivesBudget(t *testing.T) {
	// Test: with the forced planning step enabled, the very first LLM call (the planning
	// call that builds the roadmap) must include the iteration budget message so the agent
	// can size its todo list against the budget.
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			// Planning step: write the roadmap (already done to avoid a pending-todo nudge).
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{
					id:         "p1",
					name:       "TodoWrite",
					parameters: map[string]any{"items": []any{map[string]any{"id": "1", "content": "Do the thing", "status": "done"}}},
				}},
			},
			// Main loop: finish.
			{
				Message: llm.NewMessage(llm.RoleAssistant, "All planned and done."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
		WithForcePlanningStep(true),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	err = handler.Handle(context.Background(), agent.NewInput("Do the thing"), func(evt agent.Event) error {
		return nil
	})
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	if len(client.capturedMessages) == 0 {
		t.Fatal("expected at least one captured LLM call")
	}

	// The first captured call is the planning step; it must carry the budget message.
	if !messagesContainSystem(client.capturedMessages[0], "remaining to complete your task") {
		t.Error("planning step did not receive the iteration budget message")
	}
}

func TestHandler_BudgetMessageEnriched(t *testing.T) {
	// Test: the recurring budget message states the current iteration and the total, so
	// the agent knows its pace, not just a bare remaining count.
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				Message: llm.NewMessage(llm.RoleAssistant, "Done."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
		WithMaxIterations(7),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	err = handler.Handle(context.Background(), agent.NewInput("Quick question"), func(evt agent.Event) error {
		return nil
	})
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	if len(client.capturedMessages) == 0 {
		t.Fatal("expected at least one captured LLM call")
	}

	// First loop call: "Iteration 1 of 7 — 7 iterations remaining ...".
	if !messagesContainSystem(client.capturedMessages[0], "Iteration 1 of 7") {
		t.Error("budget message did not mention the current iteration and total")
	}
}

func TestHandler_AllDoneTodoAllowsCompletion(t *testing.T) {
	// Test: todo list written with all items done → LLM finishes → Complete emitted without injection.
	client := &MockChatCompletionClient{
		responses: []MockResponse{
			{
				ToolCalls: []llm.ToolCall{&MockToolCall{
					id:   "1",
					name: "TodoWrite",
					parameters: map[string]any{"items": []any{
						map[string]any{"id": "1", "content": "Task A", "status": "done"},
						map[string]any{"id": "2", "content": "Task B", "status": "done"},
					}},
				}},
			},
			{
				Message: llm.NewMessage(llm.RoleAssistant, "Everything is done."),
			},
		},
	}

	handler, err := NewHandler(
		WithClient(client),
		WithSystemPrompt("You are a helpful assistant."),
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	var events []agent.Event
	err = handler.Handle(context.Background(), agent.NewInput("Finish tasks"), func(evt agent.Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	lastEvent := events[len(events)-1]
	if lastEvent.Type() != agent.EventTypeComplete {
		t.Errorf("expected EventTypeComplete, got %s", lastEvent.Type())
	}

	if client.callCount != 2 {
		t.Errorf("expected 2 LLM calls, got %d", client.callCount)
	}
}

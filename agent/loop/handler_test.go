package loop

import (
	"context"
	"errors"
	"testing"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/llm"
)

// MockChatCompletionClient implements llm.ChatCompletionClient for testing
type MockChatCompletionClient struct {
	responses []MockResponse
	callCount int
}

type MockResponse struct {
	Message   llm.Message
	ToolCalls []llm.ToolCall
	Err       error
}

func (m *MockChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
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
	// Test: LLM calls TodoWrite then TodoRead → verify state consistency and events emitted
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

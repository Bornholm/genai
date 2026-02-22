package todo

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/llm"
)

func TestTodoWriteAndRead(t *testing.T) {
	// Test: Write then read → get back what was written
	list := NewList()
	var emittedEvents []agent.Event
	emit := func(evt agent.Event) error {
		emittedEvents = append(emittedEvents, evt)
		return nil
	}

	tools := NewTools(list, emit)

	// Find TodoWrite and TodoRead tools
	var todoWrite llm.Tool
	var todoRead llm.Tool
	for _, tool := range tools {
		if tool.Name() == "TodoWrite" {
			todoWrite = tool
		}
		if tool.Name() == "TodoRead" {
			todoRead = tool
		}
	}

	// Write items
	writeParams := map[string]any{
		"items": []any{
			map[string]any{"id": "1", "content": "Task 1", "status": "pending"},
			map[string]any{"id": "2", "content": "Task 2", "status": "in_progress"},
			map[string]any{"id": "3", "content": "Task 3", "status": "done"},
		},
	}

	result, err := todoWrite.Execute(context.Background(), writeParams)
	if err != nil {
		t.Fatalf("TodoWrite failed: %v", err)
	}

	if result.Text() != "Todo list updated successfully" {
		t.Errorf("expected success message, got: %s", result.Text())
	}

	// Verify event was emitted
	if len(emittedEvents) != 1 {
		t.Errorf("expected 1 event, got %d", len(emittedEvents))
	}
	if emittedEvents[0].Type() != agent.EventTypeTodoUpdated {
		t.Errorf("expected EventTypeTodoUpdated, got %s", emittedEvents[0].Type())
	}

	// Read items back
	readResult, err := todoRead.Execute(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("TodoRead failed: %v", err)
	}

	var items []Item
	if err := json.Unmarshal([]byte(readResult.Text()), &items); err != nil {
		t.Fatalf("failed to parse read result: %v", err)
	}

	if len(items) != 3 {
		t.Errorf("expected 3 items, got %d", len(items))
	}

	// Verify items
	if items[0].ID != "1" || items[0].Content != "Task 1" || items[0].Status != StatusPending {
		t.Errorf("item 0 mismatch: %+v", items[0])
	}
	if items[1].ID != "2" || items[1].Content != "Task 2" || items[1].Status != StatusInProgress {
		t.Errorf("item 1 mismatch: %+v", items[1])
	}
	if items[2].ID != "3" || items[2].Content != "Task 3" || items[2].Status != StatusDone {
		t.Errorf("item 2 mismatch: %+v", items[2])
	}
}

func TestTodoWriteTwice(t *testing.T) {
	// Test: Write twice → second write fully replaces first
	list := NewList()
	emit := func(evt agent.Event) error { return nil }

	tools := NewTools(list, emit)

	var todoWrite llm.Tool
	for _, tool := range tools {
		if tool.Name() == "TodoWrite" {
			todoWrite = tool
			break
		}
	}

	// First write
	writeParams1 := map[string]any{
		"items": []any{
			map[string]any{"id": "1", "content": "Task 1", "status": "pending"},
		},
	}
	_, _ = todoWrite.Execute(context.Background(), writeParams1)

	// Second write (different items)
	writeParams2 := map[string]any{
		"items": []any{
			map[string]any{"id": "2", "content": "Task 2", "status": "done"},
			map[string]any{"id": "3", "content": "Task 3", "status": "in_progress"},
		},
	}
	_, _ = todoWrite.Execute(context.Background(), writeParams2)

	// Verify list has only the second set of items
	if len(list.Items) != 2 {
		t.Errorf("expected 2 items after second write, got %d", len(list.Items))
	}

	if list.Items[0].ID != "2" {
		t.Errorf("expected first item ID to be '2', got '%s'", list.Items[0].ID)
	}
}

func TestTodoReadEmpty(t *testing.T) {
	// Test: Read with empty list → get empty array
	list := NewList()
	emit := func(evt agent.Event) error { return nil }

	tools := NewTools(list, emit)

	var todoRead llm.Tool
	for _, tool := range tools {
		if tool.Name() == "TodoRead" {
			todoRead = tool
			break
		}
	}

	result, err := todoRead.Execute(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("TodoRead failed: %v", err)
	}

	if result.Text() != "[]" {
		t.Errorf("expected '[]', got '%s'", result.Text())
	}
}

func TestTodoWriteInvalidJSON(t *testing.T) {
	// Test: Write with invalid JSON → get error result string (not Go error)
	list := NewList()
	emit := func(evt agent.Event) error { return nil }

	tools := NewTools(list, emit)

	var todoWrite llm.Tool
	for _, tool := range tools {
		if tool.Name() == "TodoWrite" {
			todoWrite = tool
			break
		}
	}

	// Missing items parameter
	result, err := todoWrite.Execute(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("expected no Go error, got: %v", err)
	}

	if result.Text() == "" {
		t.Error("expected error message in result")
	}

	// Invalid items format
	result2, err := todoWrite.Execute(context.Background(), map[string]any{
		"items": "not an array",
	})
	if err != nil {
		t.Fatalf("expected no Go error, got: %v", err)
	}

	if result2.Text() == "" {
		t.Error("expected error message in result")
	}
}

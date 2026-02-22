package todo

import (
	"context"
	"encoding/json"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/llm"
	"github.com/invopop/jsonschema"
)

// TodoWriteInput represents the input parameters for the TodoWrite tool
type TodoWriteInput struct {
	Items []TodoWriteItem `json:"items" jsonschema:"required,description=The complete list of todo items. Each item has id, content, and status (pending, in_progress, done)."`
}

// TodoWriteItem represents a single todo item for the TodoWrite tool
type TodoWriteItem struct {
	ID      string `json:"id" jsonschema:"required,description=Unique identifier for the todo item"`
	Content string `json:"content" jsonschema:"required,description=The content/description of the todo item"`
	Status  string `json:"status" jsonschema:"required,description=Status of the todo item (pending, in_progress, done),enum=pending,enum=in_progress,enum=done"`
}

// TodoReadInput represents the input parameters for the TodoRead tool (empty)
type TodoReadInput struct{}

// NewTools creates the TodoWrite and TodoRead tools
func NewTools(list *List, emit agent.EmitFunc) []llm.Tool {
	return []llm.Tool{
		NewTodoWriteTool(list, emit),
		NewTodoReadTool(list),
	}
}

// schemaToMap converts a jsonschema.Schema to map[string]any
func schemaToMap(schema *jsonschema.Schema) map[string]any {
	// Marshal the schema to JSON and unmarshal to map[string]any
	data, err := json.Marshal(schema)
	if err != nil {
		return map[string]any{}
	}
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return map[string]any{}
	}
	return result
}

// TodoWrite replaces the entire todo list
func NewTodoWriteTool(list *List, emit agent.EmitFunc) llm.Tool {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	schema := reflector.Reflect(&TodoWriteInput{})

	return llm.NewFuncTool(
		"TodoWrite",
		"Replace the entire todo list. Use this to create and update your task list. The LLM provides a full JSON array of items - this is a full replacement, not a patch.",
		schemaToMap(schema),
		func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			itemsRaw, ok := params["items"]
			if !ok {
				return llm.NewToolResult("Error: missing 'items' parameter"), nil
			}

			itemsJSON, err := json.Marshal(itemsRaw)
			if err != nil {
				return llm.NewToolResult("Error: could not serialize items"), nil
			}

			var items []TodoWriteItem

			if err := json.Unmarshal(itemsJSON, &items); err != nil {
				return llm.NewToolResult("Error: invalid items format. Expected JSON array with id, content, and status fields."), nil
			}

			// Replace the entire list
			list.Items = make([]Item, len(items))
			todoItems := make([]agent.TodoItem, len(items))
			for i, item := range items {
				status := Status(item.Status)
				if status != StatusPending && status != StatusInProgress && status != StatusDone {
					status = StatusPending
				}
				list.Items[i] = Item{
					ID:      item.ID,
					Content: item.Content,
					Status:  status,
				}
				todoItems[i] = agent.TodoItem{
					ID:      item.ID,
					Content: item.Content,
					Status:  agent.TodoStatus(status),
				}
			}

			// Emit TodoUpdated event
			if emit != nil {
				_ = emit(agent.NewEvent(agent.EventTypeTodoUpdated, &agent.TodoUpdatedData{
					Items: todoItems,
				}))
			}

			return llm.NewToolResult("Todo list updated successfully"), nil
		},
	)
}

// TodoRead returns the current todo list
func NewTodoReadTool(list *List) llm.Tool {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	schema := reflector.Reflect(&TodoReadInput{})

	return llm.NewFuncTool(
		"TodoRead",
		"Returns the current todo list as JSON.",
		schemaToMap(schema),
		func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			if len(list.Items) == 0 {
				return llm.NewToolResult("[]"), nil
			}

			data, err := json.Marshal(list.Items)
			if err != nil {
				return llm.NewToolResult("Error: could not serialize todo list"), nil
			}

			return llm.NewToolResult(string(data)), nil
		},
	)
}

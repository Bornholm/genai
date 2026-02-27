package a2a

import (
	"context"
	"log/slog"
	"sync"

	"github.com/bornholm/genai/llm"
)

// DynamicToolRegistry is a thread-safe registry that allows dynamic addition and removal of tools.
// It implements llm.Tool by delegating to registered tools.
type DynamicToolRegistry struct {
	mu    sync.RWMutex
	tools map[string]llm.Tool
}

// NewDynamicToolRegistry creates a new DynamicToolRegistry
func NewDynamicToolRegistry() *DynamicToolRegistry {
	return &DynamicToolRegistry{
		tools: make(map[string]llm.Tool),
	}
}

// AddTool adds or updates a tool in the registry
func (r *DynamicToolRegistry) AddTool(tool llm.Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[tool.Name()] = tool
	slog.Debug("dynamic tool registry: added tool", "name", tool.Name())
}

// RemoveTool removes a tool from the registry by name
func (r *DynamicToolRegistry) RemoveTool(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.tools, name)
	slog.Debug("dynamic tool registry: removed tool", "name", name)
}

// GetTool retrieves a tool by name
func (r *DynamicToolRegistry) GetTool(name string) (llm.Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tool, ok := r.tools[name]
	return tool, ok
}

// ListTools returns all registered tools
func (r *DynamicToolRegistry) ListTools() []llm.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tools := make([]llm.Tool, 0, len(r.tools))
	for _, tool := range r.tools {
		tools = append(tools, tool)
	}
	return tools
}

// ToolCount returns the number of registered tools
func (r *DynamicToolRegistry) ToolCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tools)
}

// Clear removes all tools from the registry
func (r *DynamicToolRegistry) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools = make(map[string]llm.Tool)
}

// dynamicRegistryTool is a proxy tool that looks up the actual tool at execution time
type dynamicRegistryTool struct {
	registry *DynamicToolRegistry
	name     string
}

// Name implements llm.Tool
func (t *dynamicRegistryTool) Name() string {
	return t.name
}

// Description implements llm.Tool
func (t *dynamicRegistryTool) Description() string {
	if tool, ok := t.registry.GetTool(t.name); ok {
		return tool.Description()
	}
	return "Tool not available"
}

// Parameters implements llm.Tool
func (t *dynamicRegistryTool) Parameters() map[string]any {
	if tool, ok := t.registry.GetTool(t.name); ok {
		return tool.Parameters()
	}
	return map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}
}

// Execute implements llm.Tool
func (t *dynamicRegistryTool) Execute(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
	tool, ok := t.registry.GetTool(t.name)
	if !ok {
		return llm.NewToolResult("Tool not available. The remote agent may have gone offline."), nil
	}
	return tool.Execute(ctx, params)
}

// AsTool creates a proxy tool that delegates to the registry
func (r *DynamicToolRegistry) AsTool(name string) llm.Tool {
	return &dynamicRegistryTool{
		registry: r,
		name:     name,
	}
}

// AsTools creates proxy tools for all currently registered tools
func (r *DynamicToolRegistry) AsTools() []llm.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tools := make([]llm.Tool, 0, len(r.tools))
	for name := range r.tools {
		tools = append(tools, r.AsTool(name))
	}
	return tools
}

var _ llm.Tool = &dynamicRegistryTool{}

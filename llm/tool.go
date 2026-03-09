package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/pkg/errors"
)

type Tool interface {
	Name() string
	Description() string
	Parameters() map[string]any
	Execute(ctx context.Context, params map[string]any) (ToolResult, error)
}

type ToolResult interface {
	Text() string
	Attachments() []Attachment
}

type BaseToolResult struct {
	text        string
	attachments []Attachment
}

// Attachments implements [ToolResult].
func (r *BaseToolResult) Attachments() []Attachment {
	return r.attachments
}

// Text implements [ToolResult].
func (r *BaseToolResult) Text() string {
	return r.text
}

func NewToolResult(text string, attachments ...Attachment) *BaseToolResult {
	return &BaseToolResult{
		text:        text,
		attachments: attachments,
	}
}

var _ ToolResult = &BaseToolResult{}

type FuncTool struct {
	name        string
	description string
	parameters  map[string]any
	execute     ExecuteFunc
}

type ExecuteFunc func(ctx context.Context, params map[string]any) (ToolResult, error)

// Description implements Tool.
func (f *FuncTool) Description() string {
	return f.description
}

// Execute implements Tool.
func (f *FuncTool) Execute(ctx context.Context, params map[string]any) (ToolResult, error) {
	slog.DebugContext(ctx, "executing func tool", slog.String("name", f.name), slog.Any("params", params))

	result, err := f.execute(ctx, params)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return result, nil
}

// Name implements Tool.
func (f *FuncTool) Name() string {
	return f.name
}

// Parameters implements Tool.
func (f *FuncTool) Parameters() map[string]any {
	return f.parameters
}

func NewFuncTool(name, description string, parameters map[string]any, fn ExecuteFunc) *FuncTool {
	return &FuncTool{
		name:        name,
		description: description,
		parameters:  parameters,
		execute:     fn,
	}
}

var _ Tool = &FuncTool{}

func ExecuteToolCall(ctx context.Context, tc ToolCall, tools ...Tool) (ToolMessage, error) {
	var tool Tool
	for _, t := range tools {
		if tc.Name() != t.Name() {
			continue
		}

		tool = t
		break
	}

	if tool == nil {
		toolResult := NewToolResult(fmt.Sprintf("Unknown tool named '%s'.", tc.Name()))
		return NewToolMessage(tc.ID(), toolResult), nil
	}

	var params map[string]any

	switch typ := tc.Parameters().(type) {
	case string:
		if err := json.Unmarshal([]byte(typ), &params); err != nil {
			toolResult := NewToolResult("Invalid parameter format")
			return NewToolMessage(tc.ID(), toolResult), nil
		}

	case []byte:
		if err := json.Unmarshal(typ, &params); err != nil {
			toolResult := NewToolResult("Invalid parameter format")
			return NewToolMessage(tc.ID(), toolResult), nil
		}

	case map[string]any:
		params = typ

	default:
		return nil, errors.Errorf("unexpected tool parameters type '%T'", tc.Parameters())
	}

	result, err := tool.Execute(ctx, params)
	if err != nil {
		return nil, errors.Wrap(err, "could not execute tool")
	}

	return NewToolMessage(tc.ID(), result), nil
}

func ToolParam[T any](params map[string]any, name string) (T, error) {
	raw, exists := params[name]
	if !exists {
		return *new(T), errors.Errorf("missing '%s' parameter", name)
	}

	value, ok := raw.(T)
	if !ok {
		return *new(T), errors.Errorf("invalid '%s' parameter", name)
	}

	return value, nil
}

type JSONSchema map[string]any

func NewJSONSchema() JSONSchema {
	return map[string]any{
		"type":                  "object",
		"properties":            map[string]any{},
		"required":              []string{},
		"additionalProperties": false,
	}
}

func (s JSONSchema) RequiredProperty(name, description, jsonType string, attrs ...any) JSONSchema {
	s = s.Property(name, description, jsonType, attrs...)
	required := s["required"].([]string)
	required = append(required, name)
	s["required"] = required
	return s
}

func (s JSONSchema) Property(name, description, jsonType string, attrs ...any) JSONSchema {
	properties := s["properties"].(map[string]any)

	properties[name] = map[string]any{
		"type":        jsonType,
		"description": description,
	}

	for i, v := range attrs {
		if i == 0 {
			continue
		}
		key := v.(string)
		properties[key] = attrs[i-1]
	}

	s["properties"] = properties

	return s
}

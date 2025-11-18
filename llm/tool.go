package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/pkg/errors"
)

type Tool interface {
	Name() string
	Description() string
	Parameters() map[string]any
	Execute(ctx context.Context, params map[string]any) (string, error)
}

type FuncTool struct {
	name        string
	description string
	parameters  map[string]any
	execute     ExecuteFunc
}

type ExecuteFunc func(ctx context.Context, params map[string]any) (string, error)

// Description implements Tool.
func (f *FuncTool) Description() string {
	return f.description
}

// Execute implements Tool.
func (f *FuncTool) Execute(ctx context.Context, params map[string]any) (string, error) {
	result, err := f.execute(ctx, params)
	if err != nil {
		return "", errors.WithStack(err)
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
		return NewToolMessage(tc.ID(), fmt.Sprintf("Unknown tool named '%s'.", tc.Name())), nil
	}

	var params map[string]any

	switch typ := tc.Parameters().(type) {
	case string:
		if err := json.Unmarshal([]byte(typ), &params); err != nil {
			return NewToolMessage(tc.ID(), "Invalid parameter format"), nil
		}

	case []byte:
		if err := json.Unmarshal(typ, &params); err != nil {
			return NewToolMessage(tc.ID(), "Invalid parameter format"), nil
		}

	case map[string]any:
		params = typ

	default:
		return nil, errors.Errorf("unexpected tool parameters type '%T'", tc.Parameters())
	}

	result, err := tool.Execute(ctx, params)
	if err != nil {
		return NewToolMessage(tc.ID(), fmt.Sprintf("error: %s", err.Error())), nil
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
		"additionnalProperties": false,
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

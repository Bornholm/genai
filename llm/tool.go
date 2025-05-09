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

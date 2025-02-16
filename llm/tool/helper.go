package tool

import "github.com/pkg/errors"

func Param[T any](params map[string]any, name string) (T, error) {
	raw, exists := params[name]
	if !exists {
		return *new(T), errors.Errorf("missing required parameter '%s'", name)
	}

	value, ok := raw.(T)
	if !ok {
		return *new(T), errors.Errorf("unexpected type '%T' for '%s' parameter", raw, name)
	}

	return value, nil
}

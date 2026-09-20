package codec

import (
	"encoding/json"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/types/known/structpb"
)

// parametersJSON renders tool call parameters as JSON text. It accepts the
// three forms llm.ExecuteToolCall reads (string, []byte, map) and falls back
// to json.Marshal for anything else.
func parametersJSON(params any) (string, error) {
	switch v := params.(type) {
	case nil:
		return "{}", nil
	case string:
		if v == "" {
			return "{}", nil
		}
		return v, nil
	case []byte:
		if len(v) == 0 {
			return "{}", nil
		}
		return string(v), nil
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return "", errors.WithStack(err)
		}
		return string(data), nil
	}
}

// structFromAny converts an arbitrary JSON-serializable value into a Struct
// through a JSON round trip, which is the only conversion that copes with
// typed structs, json.RawMessage and integer kinds alike.
func structFromAny(value any) (*structpb.Struct, error) {
	if value == nil {
		return nil, nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	// A typed nil (an unset map, a nil pointer) marshals to null, which is
	// no object at all.
	if string(data) == "null" {
		return nil, nil
	}
	result := &structpb.Struct{}
	if err := result.UnmarshalJSON(data); err != nil {
		return nil, errors.WithStack(err)
	}
	return result, nil
}

// valueFromAny is structFromAny for values that need not be objects.
func valueFromAny(value any) (*structpb.Value, error) {
	if value == nil {
		return nil, nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	if string(data) == "null" {
		return nil, nil
	}
	result := &structpb.Value{}
	if err := result.UnmarshalJSON(data); err != nil {
		return nil, errors.WithStack(err)
	}
	return result, nil
}

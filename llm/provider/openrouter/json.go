package openrouter

import (
	"encoding/json"

	"github.com/pkg/errors"
)

type jsonMarshaller struct {
	v any
}

// MarshalJSON implements json.Marshaler.
func (j jsonMarshaller) MarshalJSON() ([]byte, error) {
	data, err := json.Marshal(j.v)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return data, nil
}

var _ json.Marshaler = jsonMarshaller{}

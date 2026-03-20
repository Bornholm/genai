package config

import (
	"os"

	"github.com/buildkite/interpolate"
)

type envWrapper struct{}

func (e *envWrapper) Get(key string) (string, bool) {
	return os.LookupEnv(key)
}

var envLookup = &envWrapper{}

func Interpolate(s string) string {
	result, err := interpolate.Interpolate(envLookup, s)
	if err != nil {
		return s
	}
	return result
}

func InterpolateMap(m map[string]string) map[string]string {
	result := make(map[string]string, len(m))
	for k, v := range m {
		result[k] = Interpolate(v)
	}
	return result
}

func InterpolateAny(v any) any {
	switch val := v.(type) {
	case string:
		return Interpolate(val)
	case map[string]any:
		result := make(map[string]any, len(val))
		for k, v := range val {
			result[k] = InterpolateAny(v)
		}
		return result
	case []any:
		result := make([]any, len(val))
		for i, v := range val {
			result[i] = InterpolateAny(v)
		}
		return result
	default:
		return val
	}
}

package gemini

import (
	"context"
	"encoding/json"
	"log/slog"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
)

// generateContentFields lists the top-level request fields of
// generateContent that configureExtraFields does not translate: they are
// forwarded as they are, for a caller that knows the request goes to Gemini.
// A field generateContent does not define is dropped instead. Extra fields
// follow the OpenAI shape, the one the proxy forwards the fields it has no
// mapping for in, and generateContent answers 400 "Invalid JSON payload
// received. Unknown name" on any field it does not know. Clients built
// against the OpenAI API, or an OpenAI compatible gateway, send such fields
// without knowing the request will be routed to Gemini.
//
// The names are the camelCase ones the typed fields use: the API reads the
// snake_case spelling as the same field, so generation_config next to the
// typed generationConfig would be sent twice. The keys of a forwarded
// generationConfig are turned to camelCase for the same reason, before it is
// merged into the typed one.
//
// The list is the GenerateContentRequest of the Gemini API reference
// (ai.google.dev/api/generate-content, generativelanguage v1beta) as of
// October 2026; the Vertex AI request differs.
//
// A field forwarded this way overrides the one buildRequest sets from a typed
// option of the same name, except generationConfig, merged key by key into
// the typed one. A forwarded tools or toolConfig is not read: a tool_choice
// naming a function is checked against the typed tools only, so combining it
// with a forwarded tools is not supported.
//
// store and generationConfig are listed as fields generateContent defines,
// but configureExtraFields checks them before forwarding them.
var generateContentFields = map[string]struct{}{
	"cachedContent":     {},
	"contents":          {},
	"continuationToken": {},
	"generationConfig":  {},
	"labels":            {},
	"safetySettings":    {},
	"serviceTier":       {},
	"store":             {},
	"systemInstruction": {},
	"toolConfig":        {},
	"tools":             {},
}

// maxStopSequences is the most stopSequences generateContent accepts.
const maxStopSequences = 5

// configureExtraFields sets the generateContent counterpart of the OpenAI
// fields that have one, forwards the fields generateContent defines (see
// generateContentFields) and drops the rest. Each field dropped, in whole or
// in part, is logged at debug level, except the defaults it consumes: a zero
// penalty or top_k and parallel_tool_calls true. It runs after the typed options.
//
//   - stop becomes generationConfig.stopSequences, without empty entries and
//     cut to the first maxStopSequences.
//   - top_p, top_k, frequency_penalty and presence_penalty become topP, topK,
//     frequencyPenalty and presencePenalty in generationConfig. A top_p
//     outside [0, 1] is dropped; the other ranges depend on the model and
//     are left to the API. A zero penalty, the default of both APIs that
//     OpenAI clients often send, is left out: some Gemini models refuse any
//     penalty. So is a zero top_k, which gateways send to mean unset: models
//     running with nucleus sampling refuse any topK.
//   - the object form of tool_choice, which names a function, forces that
//     function: mode ANY with it as the only allowed function name. It must
//     be one of the request's tools. The string forms are dropped: the proxy
//     maps them to llm.ToolChoice, the option a library caller sets instead.
//   - parallel_tool_calls has no counterpart and is dropped: Gemini may still
//     call several functions at once.
//   - store shares its name and its boolean with OpenAI's field, and on
//     Gemini overrides the project's logging setting. Only false is
//     forwarded: a client may ask for its request not to be kept, not turn
//     on a logging the operator left off.
//   - a forwarded generationConfig is merged into the typed one with its
//     keys in camelCase. When it spells a key several ways, the camelCase one
//     wins, or else the first in sorted order, and the others are dropped.
//   - every other field, OpenAI's user, logprobs or service_tier and the
//     fields of OpenAI compatible gateways and servers alike, is dropped.
func configureExtraFields(ctx context.Context, req *request, config *generationConfig, extra map[string]any) {
	if len(extra) == 0 {
		return
	}

	remaining := make(map[string]any, len(extra))
	var dropped []string
	drop := func(key string) { dropped = append(dropped, key) }

	for key, value := range extra {
		switch key {
		case "stop":
			stop, ok := stringList(value)
			if !ok {
				drop(key)
				continue
			}
			if len(stop) > maxStopSequences {
				stop = stop[:maxStopSequences]
				drop(key + "[" + strconv.Itoa(maxStopSequences) + ":]")
			}
			config.StopSequences = stop

		case "top_p":
			topP, ok := asFloat(value)
			if !ok || topP < 0 || topP > 1 {
				drop(key)
				continue
			}
			config.TopP = &topP

		case "top_k":
			// The upper bound keeps the int conversion defined.
			topK, ok := asFloat(value)
			if !ok || topK < 0 || topK > math.MaxInt32 || topK != math.Trunc(topK) {
				drop(key)
				continue
			}
			if topK == 0 {
				continue
			}
			k := int(topK)
			config.TopK = &k

		case "frequency_penalty", "presence_penalty":
			penalty, ok := asFloat(value)
			if !ok {
				drop(key)
				continue
			}
			if penalty == 0 {
				continue
			}
			if key == "frequency_penalty" {
				config.FrequencyPenalty = &penalty
			} else {
				config.PresencePenalty = &penalty
			}

		case "tool_choice":
			if !forceFunction(req, value) {
				drop(key)
			}

		case "parallel_tool_calls":
			if parallel, ok := value.(bool); !ok || !parallel {
				drop(key)
			}

		case "store":
			if store, ok := value.(bool); !ok || store {
				drop(key)
				continue
			}
			remaining[key] = value

		case "generationConfig":
			// Merged into the typed one: anything but an object would
			// replace it.
			forwarded, ok := value.(map[string]any)
			if !ok {
				drop(key)
				continue
			}
			// Sorted, so that two snake_case spellings of one field, top_p
			// and top__p, resolve the same way on every run.
			camel := make(map[string]any, len(forwarded))
			for _, field := range slices.Sorted(maps.Keys(forwarded)) {
				name := camelCase(field)
				if name != field {
					_, spelledCamel := forwarded[name]
					_, seen := camel[name]
					if spelledCamel || seen {
						drop(key + "." + field)
						continue
					}
				}
				camel[name] = forwarded[field]
			}
			remaining[key] = camel

		default:
			if _, defined := generateContentFields[key]; !defined {
				drop(key)
				continue
			}
			remaining[key] = value
		}
	}

	if len(dropped) > 0 {
		slices.Sort(dropped)
		slog.DebugContext(ctx, "dropped request fields the gemini generatecontent api does not accept", slog.Any("fields", dropped))
	}

	if len(remaining) > 0 {
		forwarded := slices.Sorted(maps.Keys(remaining))
		slog.DebugContext(ctx, "forwarding request fields verbatim to the gemini generatecontent api", slog.Any("fields", forwarded))
		req.extra = remaining
	}
}

// forceFunction forces the function OpenAI's tool_choice object names,
// {"type":"function","function":{"name":...}}, and reports whether it could:
// the function must be one of the request's tools.
func forceFunction(req *request, value any) bool {
	choice, ok := value.(map[string]any)
	if !ok || choice["type"] != "function" {
		return false
	}
	function, _ := choice["function"].(map[string]any)
	name, _ := function["name"].(string)
	if !declaresFunction(req, name) {
		return false
	}
	req.ToolConfig = &toolConfig{FunctionCallingConfig: functionCallingConfig{
		Mode:                 "ANY",
		AllowedFunctionNames: []string{name},
	}}
	return true
}

// declaresFunction reports whether the request declares a function of that
// name.
func declaresFunction(req *request, name string) bool {
	if name == "" {
		return false
	}
	for _, set := range req.Tools {
		for _, declaration := range set.FunctionDeclarations {
			if declaration.Name == name {
				return true
			}
		}
	}
	return false
}

// camelCase turns a snake_case field name, top_p for instance, into the
// camelCase spelling the API also reads, topP. A camelCase name is left as is.
func camelCase(name string) string {
	parts := strings.Split(name, "_")
	for i := 1; i < len(parts); i++ {
		if parts[i] != "" {
			parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
		}
	}
	return strings.Join(parts, "")
}

// stringList reads OpenAI's stop, a string or an array of strings. Empty
// strings are left out.
func stringList(value any) ([]string, bool) {
	var items []any
	switch v := value.(type) {
	case string:
		items = []any{v}
	case []string:
		for _, item := range v {
			items = append(items, item)
		}
	case []any:
		items = v
	default:
		return nil, false
	}

	list := make([]string, 0, len(items))
	for _, item := range items {
		s, ok := item.(string)
		if !ok {
			return nil, false
		}
		if s != "" {
			list = append(list, s)
		}
	}
	return list, len(list) > 0
}

// asFloat reads a number of any Go numeric type or a json.Number. NaN and the
// infinities are refused: JSON cannot carry them, and the request would fail
// to marshal.
func asFloat(value any) (float64, bool) {
	var f float64
	switch v := value.(type) {
	case float64:
		f = v
	case float32:
		f = float64(v)
	case int:
		f = float64(v)
	case int64:
		f = float64(v)
	case int32:
		f = float64(v)
	case int16:
		f = float64(v)
	case int8:
		f = float64(v)
	case uint:
		f = float64(v)
	case uint64:
		f = float64(v)
	case uint32:
		f = float64(v)
	case uint16:
		f = float64(v)
	case uint8:
		f = float64(v)
	case json.Number:
		var err error
		if f, err = v.Float64(); err != nil {
			return 0, false
		}
	default:
		return 0, false
	}
	return f, !math.IsNaN(f) && !math.IsInf(f, 0)
}

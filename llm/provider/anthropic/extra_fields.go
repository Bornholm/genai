package anthropic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"maps"
	"slices"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
)

// openAIOnlyFields lists OpenAI request fields the Messages API has no
// counterpart for. Extra fields follow the OpenAI shape, the one the proxy
// forwards the fields it has no mapping for in, so that portable sampling
// parameters are not silently dropped. The Messages API answers 400 "Extra
// inputs are not permitted" on any field it does not know, and clients built
// against the OpenAI API send these without knowing the request will be routed
// to Anthropic.
var openAIOnlyFields = map[string]struct{}{
	"frequency_penalty":  {},
	"presence_penalty":   {},
	"logit_bias":         {},
	"logprobs":           {},
	"top_logprobs":       {},
	"n":                  {},
	"seed":               {},
	"modalities":         {},
	"audio":              {},
	"prediction":         {},
	"store":              {},
	"prompt_cache_key":   {},
	"safety_identifier":  {},
	"web_search_options": {},
	"verbosity":          {},
	"stream_options":     {},
	"functions":          {},
	"function_call":      {},
	// The proxy maps these to typed options, but a caller setting them as
	// extra fields would send them as they are.
	"max_completion_tokens": {},
	"reasoning_effort":      {},
	"response_format":       {},
}

// configureExtraFields sets the Messages API counterpart of the OpenAI fields
// that have one, drops the ones that have none, and forwards the rest verbatim,
// such as an Anthropic field like top_k set by a caller that knows the request
// goes to Anthropic. It runs after the typed options, whose rules it follows:
// thinking is whether buildParams enabled it.
//
//   - stop becomes stop_sequences.
//   - user becomes metadata.user_id, as a fingerprint: the API asks for an
//     opaque identifier, and OpenAI's user is often an email or a login. The
//     fingerprint has the format of the proxy's TokenFingerprint. A
//     metadata.user_id the caller sets is sent as it is.
//   - top_p is kept unless thinking is on, where the API restricts it, as the
//     temperature is. A model refusing it next to a temperature answers 400:
//     the caller sent both.
//   - the object form of tool_choice sets the tool choice: OpenAI's, which
//     names a function, or the Messages API's own. A named tool must be one
//     of the request's tools. Under thinking, forcing a tool falls back to
//     auto, as ToolChoiceRequired does.
//   - parallel_tool_calls false disables parallel tool use.
//   - top_k, an Anthropic field, is forwarded unless thinking is on, where the
//     API refuses it.
//   - metadata and service_tier exist on both APIs with different values.
//     Only metadata.user_id is kept, OpenAI's other tags dropped, and
//     service_tier only with a value the Messages API accepts.
func configureExtraFields(ctx context.Context, params *anthropicsdk.MessageNewParams, extra map[string]any, thinking bool) {
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
			params.StopSequences = stop

		case "user":
			user, ok := value.(string)
			if !ok || user == "" {
				drop(key)
				continue
			}
			if !params.Metadata.UserID.Valid() {
				params.Metadata.UserID = anthropicsdk.String(userFingerprint(user))
			}

		case "metadata":
			metadata, _ := value.(map[string]any)
			for tag := range metadata {
				if tag != "user_id" {
					drop(key + "." + tag)
				}
			}
			userID := asString(metadata["user_id"])
			if userID == "" {
				if len(metadata) == 0 {
					drop(key)
				}
				continue
			}
			params.Metadata.UserID = anthropicsdk.String(userID)

		case "top_k":
			if thinking {
				drop(key)
				continue
			}
			remaining[key] = value

		case "top_p":
			topP, ok := asFloat(value)
			if !ok || thinking {
				drop(key)
				continue
			}
			params.TopP = anthropicsdk.Float(topP)

		case "service_tier":
			switch tier := anthropicsdk.MessageNewParamsServiceTier(asString(value)); tier {
			case anthropicsdk.MessageNewParamsServiceTierAuto, anthropicsdk.MessageNewParamsServiceTierStandardOnly:
				params.ServiceTier = tier
			default:
				drop(key)
			}

		case "tool_choice", "parallel_tool_calls":
			// Applied below, once the tool choice they refine is known.

		default:
			if _, openAIOnly := openAIOnlyFields[key]; openAIOnly {
				drop(key)
				continue
			}
			remaining[key] = value
		}
	}

	if !configureToolChoice(params, extra["tool_choice"], thinking) {
		if _, set := extra["tool_choice"]; set {
			drop("tool_choice")
		}
	}
	if parallel, set := extra["parallel_tool_calls"]; set && !disableParallelToolUse(params, parallel) {
		drop("parallel_tool_calls")
	}

	if len(dropped) > 0 {
		slices.Sort(dropped)
		slog.DebugContext(ctx, "dropped request fields the anthropic messages api does not accept", slog.Any("fields", dropped))
	}

	if len(remaining) > 0 {
		// A field missing from the lists above reaches the API as is, and a
		// 400 on it is told apart from the logs.
		forwarded := slices.Sorted(maps.Keys(remaining))
		slog.DebugContext(ctx, "forwarding request fields verbatim to the anthropic messages api", slog.Any("fields", forwarded))
		params.SetExtraFields(remaining)
	}
}

// configureToolChoice sets the tool choice a tool_choice object gives, and
// reports whether it could. OpenAI's object names a function,
// {"type":"function","function":{"name":...}}; the Messages API's own is
// {"type":"auto"|"any"|"none"} or {"type":"tool","name":...}. The string forms
// reach buildParams as llm.ToolChoice and are not seen here. A tool the request
// does not declare is refused, which leaves the tool choice buildParams set.
func configureToolChoice(params *anthropicsdk.MessageNewParams, value any, thinking bool) bool {
	choice, ok := value.(map[string]any)
	if !ok || len(params.Tools) == 0 {
		return false
	}

	var disable param.Opt[bool]
	if d, ok := choice["disable_parallel_tool_use"].(bool); ok {
		disable = anthropicsdk.Bool(d)
	}
	auto := anthropicsdk.ToolChoiceUnionParam{OfAuto: &anthropicsdk.ToolChoiceAutoParam{DisableParallelToolUse: disable}}

	switch choice["type"] {
	case "auto":
		params.ToolChoice = auto
	case "none":
		params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfNone: &anthropicsdk.ToolChoiceNoneParam{}}
	case "any":
		// With extended thinking the API only accepts auto or none.
		if thinking {
			params.ToolChoice = auto
			return true
		}
		params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfAny: &anthropicsdk.ToolChoiceAnyParam{DisableParallelToolUse: disable}}
	case "function", "tool":
		name := asString(choice["name"])
		if function, ok := choice["function"].(map[string]any); ok {
			name = asString(function["name"])
		}
		if !declaresTool(params, name) {
			return false
		}
		if thinking {
			params.ToolChoice = auto
			return true
		}
		params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfTool: &anthropicsdk.ToolChoiceToolParam{Name: name, DisableParallelToolUse: disable}}
	default:
		return false
	}
	return true
}

// declaresTool reports whether the request declares a tool of that name.
func declaresTool(params *anthropicsdk.MessageNewParams, name string) bool {
	if name == "" {
		return false
	}
	for _, tool := range params.Tools {
		if tool.OfTool != nil && tool.OfTool.Name == name {
			return true
		}
	}
	return false
}

// disableParallelToolUse applies parallel_tool_calls false to the tool choice,
// and reports whether the value could be honoured. True is the default of both
// APIs. A request without tools, or one forbidding them, has nothing to
// disable.
func disableParallelToolUse(params *anthropicsdk.MessageNewParams, value any) bool {
	parallel, ok := value.(bool)
	if !ok {
		return false
	}
	if parallel || len(params.Tools) == 0 {
		return true
	}

	disable := anthropicsdk.Bool(true)
	switch choice := &params.ToolChoice; {
	case choice.OfAuto != nil:
		choice.OfAuto.DisableParallelToolUse = disable
	case choice.OfAny != nil:
		choice.OfAny.DisableParallelToolUse = disable
	case choice.OfTool != nil:
		choice.OfTool.DisableParallelToolUse = disable
	case choice.OfNone != nil:
	default:
		// No tool choice: the API's default is auto.
		choice.OfAuto = &anthropicsdk.ToolChoiceAutoParam{DisableParallelToolUse: disable}
	}
	return true
}

// stringList reads OpenAI's stop, a string or an array of strings. Empty
// strings are left out, the Messages API refusing an empty stop sequence.
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

// userFingerprint is "sha256:" followed by the first 16 hexadecimal digits of
// the SHA-256 of user, the format of the proxy's TokenFingerprint. It keeps
// the identifier stable without sending it: a guessable one, such as an email,
// is pseudonymised rather than protected.
func userFingerprint(user string) string {
	sum := sha256.Sum256([]byte(user))
	return "sha256:" + hex.EncodeToString(sum[:8])
}

func asFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case int32:
		return float64(v), true
	case json.Number:
		f, err := v.Float64()
		return f, err == nil
	}
	return 0, false
}

func asString(value any) string {
	s, _ := value.(string)
	return s
}

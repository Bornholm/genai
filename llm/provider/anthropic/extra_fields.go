package anthropic

import (
	"log/slog"
	"slices"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
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
}

// configureExtraFields sets the Messages API counterpart of the OpenAI fields
// that have one, drops the ones that have none, and forwards the rest verbatim,
// such as an Anthropic field like top_k set by a caller that knows the request
// goes to Anthropic. It runs after the typed options, whose rules it follows:
// thinking is whether buildParams enabled it.
//
//   - stop becomes stop_sequences.
//   - user becomes metadata.user_id.
//   - top_p is kept unless thinking is on or a temperature is set: the API
//     restricts it under thinking, and some models refuse both together.
//   - the object form of tool_choice, which names a function, forces that
//     tool, or falls back to auto under thinking as ToolChoiceRequired does.
//   - parallel_tool_calls false disables parallel tool use.
//   - metadata and service_tier exist on both APIs with different values, so
//     they are kept only when the Messages API accepts them.
func configureExtraFields(params *anthropicsdk.MessageNewParams, extra map[string]any, thinking bool) {
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
				params.Metadata.UserID = anthropicsdk.String(user)
			}

		case "metadata":
			userID, ok := metadataUserID(value)
			if !ok {
				drop(key)
				continue
			}
			params.Metadata.UserID = anthropicsdk.String(userID)

		case "top_p":
			topP, ok := asFloat(value)
			if !ok || thinking || params.Temperature.Valid() {
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
		slog.Debug("dropped request fields the anthropic messages api does not accept", slog.Any("fields", dropped))
	}

	if len(remaining) > 0 {
		params.SetExtraFields(remaining)
	}
}

// configureToolChoice forces the function an OpenAI tool_choice object names,
// {"type":"function","function":{"name":...}}, and reports whether it did.
// The string forms reach buildParams as llm.ToolChoice and are not seen here.
func configureToolChoice(params *anthropicsdk.MessageNewParams, value any, thinking bool) bool {
	choice, ok := value.(map[string]any)
	if !ok || choice["type"] != "function" || len(params.Tools) == 0 {
		return false
	}
	function, _ := choice["function"].(map[string]any)
	name := asString(function["name"])
	if name == "" {
		return false
	}
	// With extended thinking the API only accepts auto or none.
	if thinking {
		params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfAuto: &anthropicsdk.ToolChoiceAutoParam{}}
		return true
	}
	params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfTool: &anthropicsdk.ToolChoiceToolParam{Name: name}}
	return true
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

// stringList reads OpenAI's stop, a string or an array of strings.
func stringList(value any) ([]string, bool) {
	switch v := value.(type) {
	case string:
		if v == "" {
			return nil, false
		}
		return []string{v}, true
	case []string:
		return v, len(v) > 0
	case []any:
		list := make([]string, 0, len(v))
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				return nil, false
			}
			list = append(list, s)
		}
		return list, len(list) > 0
	}
	return nil, false
}

// metadataUserID reads a metadata object the Messages API accepts, one whose
// only field is user_id. OpenAI's metadata is a map of arbitrary tags.
func metadataUserID(value any) (string, bool) {
	metadata, ok := value.(map[string]any)
	if !ok || len(metadata) != 1 {
		return "", false
	}
	userID := asString(metadata["user_id"])
	return userID, userID != ""
}

func asFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	}
	return 0, false
}

func asString(value any) string {
	s, _ := value.(string)
	return s
}

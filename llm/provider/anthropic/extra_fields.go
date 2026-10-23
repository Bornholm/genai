package anthropic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"maps"
	"math"
	"slices"
	"unicode/utf8"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
)

// messagesAPIFields lists the request fields of the Messages API, beta ones
// included, that configureExtraFields does not translate: they are forwarded
// as they are, for a caller that knows the request goes to Anthropic. The
// provider calls the stable endpoint: a beta field, one BetaMessageNewParams
// alone defines, is accepted only with the anthropic-beta header that enables
// it, which the provider's Headers option sets. A field
// the API does not define is dropped instead. Extra fields follow the OpenAI
// shape, the one the proxy forwards the fields it has no mapping for in, and
// the Messages API answers 400 "Extra inputs are not permitted" on any field
// it does not know. Clients built against the OpenAI API, or an OpenAI
// compatible gateway, send such fields without knowing the request will be
// routed to Anthropic.
//
// A field forwarded this way overrides the one buildParams sets from a typed
// option of the same name. A forwarded thinking is read, so the sampling and
// tool choice rules follow it, one way only: one enabling thinking drops what
// the API refuses with it, one disabling it does not bring back the
// temperature or forced tool choice a typed reasoning option dropped. A
// forwarded max_tokens or tools is not read: the caller then answers for a
// thinking budget under max_tokens, and tool_choice and parallel_tool_calls
// are applied against the typed tools only, so with tools forwarded and none
// typed both are dropped.
//
// The list was checked against MessageNewParams and BetaMessageNewParams of
// anthropic-sdk-go v1.72.0; TestMessagesAPIFields_CoversTheSDKParams fails
// when a bump adds a field to either.
var messagesAPIFields = map[string]struct{}{
	"cache_control":         {},
	"container":             {},
	"context_management":    {},
	"diagnostics":           {},
	"fallback_credit_token": {},
	"fallbacks":             {},
	"inference_geo":         {},
	"max_tokens":            {},
	"mcp_servers":           {},
	"messages":              {},
	"model":                 {},
	"output_config":         {},
	"output_format":         {},
	"speed":                 {},
	"stop_sequences":        {},
	"system":                {},
	"thinking":              {},
	"tools":                 {},
}

// maxUserIDLength is the longest metadata.user_id the Messages API accepts.
const maxUserIDLength = 256

// configureExtraFields sets the Messages API counterpart of the OpenAI fields
// that have one, forwards the fields the Messages API defines (see
// messagesAPIFields) and drops the rest. Each field dropped, in whole or in
// part, is logged at debug level. It runs after the typed options, whose rules
// it follows: thinking is whether buildParams enabled it, unless a forwarded
// thinking field says otherwise.
//
//   - stop becomes stop_sequences.
//   - user becomes metadata.user_id, as a fingerprint: the API asks for an
//     opaque identifier, and OpenAI's user is often an email or a login. The
//     fingerprint has the format of the proxy's TokenFingerprint. A
//     metadata.user_id the caller sets wins over user and is sent as it is,
//     cut to the length the API accepts.
//   - thinking, forwarded as it is, turns the rules below on or off. When it
//     enables thinking, the temperature, set by an option or forwarded, is
//     dropped and a forced tool choice falls back to auto, as with an
//     option enabling it.
//   - temperature is capped at maxTemperature, the OpenAI range going up to
//     2, and dropped under thinking, as the typed option is.
//   - top_p is kept unless thinking is on, where the API restricts it, as the
//     temperature is. A model refusing it next to a temperature answers 400:
//     the caller sent both.
//   - the object form of tool_choice sets the tool choice: OpenAI's, which
//     names a function, or the Messages API's own. A named tool must be one
//     of the request's tools. Under thinking, forcing a tool falls back to
//     auto, as ToolChoiceRequired does.
//   - parallel_tool_calls false disables parallel tool use, when the request
//     has tools it may call.
//   - the string form of tool_choice is dropped: the proxy maps it to
//     llm.ToolChoice, the option a library caller sets instead.
//   - top_k, an Anthropic field, is forwarded when it is a whole number,
//     unless thinking is on, where the API refuses it.
//   - a native field and its OpenAI counterpart, stop_sequences and stop for
//     instance, are both read; the native one, forwarded last, wins.
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
	var user, metadataUserID string

	if on, set := forwardedThinking(extra["thinking"]); set && on != thinking {
		thinking = on
		if thinking {
			if params.Temperature.Valid() {
				params.Temperature = param.Opt[float64]{}
				drop("temperature")
			}
			unforceToolChoice(params)
		}
	}

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
			// Applied below, once metadata.user_id, which wins, is known.
			user = asString(value)
			if user == "" {
				drop(key)
			}

		case "metadata":
			metadata, ok := value.(map[string]any)
			if !ok {
				drop(key)
				continue
			}
			for tag, tagValue := range metadata {
				if tag != "user_id" {
					drop(key + "." + tag)
					continue
				}
				metadataUserID = asString(tagValue)
				if metadataUserID == "" {
					drop(key + "." + tag)
				}
			}

		case "top_k":
			topK, ok := asFloat(value)
			// The upper bound keeps the int64 conversion defined.
			if !ok || topK < 0 || topK > math.MaxInt32 || topK != math.Trunc(topK) || thinking {
				drop(key)
				continue
			}
			params.TopK = anthropicsdk.Int(int64(topK))

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

		case "temperature":
			// As buildParams has it: the API rejects any temperature but the
			// default once thinking is on, and stops at maxTemperature where
			// the OpenAI range goes up to 2.
			temperature, ok := asFloat(value)
			if !ok || thinking {
				drop(key)
				continue
			}
			params.Temperature = anthropicsdk.Float(min(temperature, maxTemperature))

		default:
			if _, defined := messagesAPIFields[key]; !defined {
				drop(key)
				continue
			}
			remaining[key] = value
		}
	}

	switch {
	case metadataUserID != "":
		if user != "" {
			drop("user")
		}
		params.Metadata.UserID = anthropicsdk.String(truncate(metadataUserID, maxUserIDLength))
	case user != "":
		params.Metadata.UserID = anthropicsdk.String(userFingerprint(user))
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
		dropped = slices.Compact(dropped)
		slog.DebugContext(ctx, "dropped request fields the anthropic messages api does not accept", slog.Any("fields", dropped))
	}

	if len(remaining) > 0 {
		forwarded := slices.Sorted(maps.Keys(remaining))
		slog.DebugContext(ctx, "forwarding request fields verbatim to the anthropic messages api", slog.Any("fields", forwarded))
		params.SetExtraFields(remaining)
	}
}

// configureToolChoice sets the tool choice a tool_choice object gives, and
// reports whether it could. OpenAI's object names a function,
// {"type":"function","function":{"name":...}}; the Messages API's own is
// {"type":"auto"|"any"|"none"} or {"type":"tool","name":...}. The string forms
// are not read here. A tool the request does not declare is refused, which
// leaves the tool choice buildParams set.
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

// forwardedThinking reads a thinking field forwarded as it is, and reports
// whether it enables thinking and whether it says.
func forwardedThinking(value any) (on, set bool) {
	config, ok := value.(map[string]any)
	if !ok {
		return false, false
	}
	switch config["type"] {
	case "enabled", "adaptive":
		return true, true
	case "disabled":
		return false, true
	}
	return false, false
}

// unforceToolChoice turns a tool choice forcing a tool into auto, the one
// besides none the API accepts with thinking on, keeping its parallel tool use.
func unforceToolChoice(params *anthropicsdk.MessageNewParams) {
	var disable param.Opt[bool]
	switch choice := params.ToolChoice; {
	case choice.OfAny != nil:
		disable = choice.OfAny.DisableParallelToolUse
	case choice.OfTool != nil:
		disable = choice.OfTool.DisableParallelToolUse
	default:
		return
	}
	params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfAuto: &anthropicsdk.ToolChoiceAutoParam{DisableParallelToolUse: disable}}
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

// disableParallelToolUse applies parallel_tool_calls to the tool choice, and
// reports whether the value had an effect to have. True is the default of both
// APIs, consumed without being forwarded or logged. False needs tools the request may call: without tools, or with tools
// forbidden, there is nothing to disable.
func disableParallelToolUse(params *anthropicsdk.MessageNewParams, value any) bool {
	parallel, ok := value.(bool)
	if !ok {
		return false
	}
	if parallel {
		return true
	}
	if len(params.Tools) == 0 {
		return false
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
		return false
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
// is pseudonymised rather than protected. Keep it in step with
// proxy.TokenFingerprint, which this package cannot import: a proxy test
// checks the two agree.
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

// truncate cuts s to at most n characters.
func truncate(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	return string([]rune(s)[:n])
}

func asString(value any) string {
	s, _ := value.(string)
	return s
}

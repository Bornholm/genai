package anthropic

import (
	"encoding/json"
	"fmt"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// buildMessages converts genai messages to the Messages API shape: system
// prompts are hoisted into the dedicated system field, wherever they sit in
// the conversation, and the rest is folded into strictly alternating
// user/assistant turns.
//
// Folding matters beyond cosmetics: a tool_result must live in the user
// turn that immediately follows the assistant turn carrying its tool_use,
// so consecutive tool messages have to share one user turn.
//
// A cache hint means "cache everything up to and including this message".
// Turns being reshuffled by the folding, the breakpoints are placed once
// the turns are final, each on the block its message ended with; the API
// allows maxCacheBreakpoints of them per request, system blocks included.
// The cached prefix is defined by the wire order of the final turn, so a
// tool result folded after an annotated user text ends up inside the
// prefix, ahead of it.
func buildMessages(msgs []llm.Message) ([]anthropicsdk.TextBlockParam, []anthropicsdk.MessageParam, error) {
	var (
		system      []anthropicsdk.TextBlockParam
		messages    []anthropicsdk.MessageParam
		breakpoints []cacheBreakpoint
	)

	add := func(role anthropicsdk.MessageParamRole, blocks []anthropicsdk.ContentBlockParamUnion, cc *anthropicsdk.CacheControlEphemeralParam) {
		if len(blocks) == 0 {
			// Nothing to send for this message; its cache hint still means
			// "cache everything so far", so it moves to the end of the
			// previous turn. With no previous turn nothing precedes it and
			// the hint is void.
			if cc != nil && len(messages) > 0 {
				turn := &messages[len(messages)-1]
				breakpoints = append(breakpoints, cacheBreakpoint{turn: len(messages) - 1, block: turn.Content[len(turn.Content)-1], cc: cc})
			}
			return
		}
		last := blocks[len(blocks)-1]
		if n := len(messages); n > 0 && messages[n-1].Role == role {
			messages[n-1].Content = orderTurn(role, append(messages[n-1].Content, blocks...))
		} else {
			messages = append(messages, anthropicsdk.MessageParam{Role: role, Content: blocks})
		}
		if cc != nil {
			breakpoints = append(breakpoints, cacheBreakpoint{turn: len(messages) - 1, block: last, cc: cc})
		}
	}

	for _, m := range msgs {
		cc, err := cacheControl(m)
		if err != nil {
			return nil, nil, errors.WithStack(err)
		}

		switch m.Role() {
		case llm.RoleSystem:
			if len(m.Attachments()) > 0 {
				return nil, nil, errors.New("system messages cannot have attachments")
			}
			if m.Content() == "" {
				// An empty text block is refused by the API; the message
				// carries nothing but, possibly, a cache hint, which then
				// covers the system blocks so far.
				if cc != nil && len(system) > 0 {
					system[len(system)-1].CacheControl = *cc
				}
				continue
			}
			block := anthropicsdk.TextBlockParam{Text: m.Content()}
			if cc != nil {
				block.CacheControl = *cc
			}
			system = append(system, block)

		case llm.RoleUser:
			blocks, err := userBlocks(m)
			if err != nil {
				return nil, nil, errors.WithStack(err)
			}
			add(anthropicsdk.MessageParamRoleUser, blocks, cc)

		case llm.RoleAssistant:
			if len(m.Attachments()) > 0 {
				return nil, nil, errors.New("assistant messages cannot have attachments")
			}
			blocks := reasoningBlocks(m)
			if content := m.Content(); content != "" {
				blocks = append(blocks, anthropicsdk.NewTextBlock(content))
			}
			if len(blocks) == 0 {
				// Nothing to replay, whether the message is empty or only
				// carries unsigned reasoning from another provider:
				// dropping the turn would silently fold the user turns
				// around it and hand the model a different history.
				if hasReasoning(m) {
					return nil, nil, llm.NewValidationError("messages", "assistant message carries only unsigned reasoning, which the Messages API cannot replay")
				}
				return nil, nil, llm.NewValidationError("messages", "assistant message has no content to send")
			}
			add(anthropicsdk.MessageParamRoleAssistant, blocks, cc)

		case llm.RoleToolCalls:
			if len(m.Attachments()) > 0 {
				return nil, nil, errors.New("tool calls messages cannot have attachments")
			}
			toolCallsMessage, ok := m.(llm.ToolCallsMessage)
			if !ok {
				return nil, nil, errors.Errorf("unexpected tool calls message type '%T'", m)
			}
			blocks := reasoningBlocks(m)
			if content := m.Content(); content != "" {
				blocks = append(blocks, anthropicsdk.NewTextBlock(content))
			}
			for _, tc := range toolCallsMessage.ToolCalls() {
				input, err := toolCallInput(tc.Parameters())
				if err != nil {
					return nil, nil, errors.Wrapf(err, "invalid parameters for tool call '%s'", tc.ID())
				}
				blocks = append(blocks, anthropicsdk.NewToolUseBlock(tc.ID(), input, tc.Name()))
			}
			add(anthropicsdk.MessageParamRoleAssistant, blocks, cc)

		case llm.RoleTool:
			toolMessage, ok := m.(llm.ToolMessage)
			if !ok {
				return nil, nil, errors.Errorf("unexpected tool message type '%T'", m)
			}
			result := anthropicsdk.ToolResultBlockParam{ToolUseID: toolMessage.ID()}
			if content := m.Content(); content != "" {
				result.Content = append(result.Content, anthropicsdk.ToolResultBlockParamContentUnion{
					OfText: &anthropicsdk.TextBlockParam{Text: content},
				})
			}
			for _, attachment := range m.Attachments() {
				part, err := toolResultBlockContent(attachment)
				if err != nil {
					return nil, nil, errors.WithStack(err)
				}
				result.Content = append(result.Content, part)
			}
			add(anthropicsdk.MessageParamRoleUser, []anthropicsdk.ContentBlockParamUnion{{OfToolResult: &result}}, cc)

		default:
			return nil, nil, errors.Errorf("unsupported message role '%s'", m.Role())
		}
	}

	for _, bp := range breakpoints {
		content := messages[bp.turn].Content
		for i := range content {
			if sameBlock(content[i], bp.block) {
				applyCacheControl(content[i:i+1], bp.cc)
				break
			}
		}
	}

	// Counted once placed: hints that resolved on the same block, or on a
	// thinking block that cannot carry one, produce no marker on the wire.
	count := 0
	for _, block := range system {
		if block.CacheControl.Type != "" {
			count++
		}
	}
	for _, turn := range messages {
		for _, block := range turn.Content {
			if hasCacheControl(block) {
				count++
			}
		}
	}
	if count > maxCacheBreakpoints {
		return nil, nil, llm.NewValidationError("cache_control", fmt.Sprintf("at most %d cache breakpoints per request, got %d", maxCacheBreakpoints, count))
	}

	return system, messages, nil
}

// hasCacheControl reports whether a block carries a cache marker.
func hasCacheControl(b anthropicsdk.ContentBlockParamUnion) bool {
	switch {
	case b.OfText != nil:
		return b.OfText.CacheControl.Type != ""
	case b.OfImage != nil:
		return b.OfImage.CacheControl.Type != ""
	case b.OfDocument != nil:
		return b.OfDocument.CacheControl.Type != ""
	case b.OfToolUse != nil:
		return b.OfToolUse.CacheControl.Type != ""
	case b.OfToolResult != nil:
		return b.OfToolResult.CacheControl.Type != ""
	}
	return false
}

// maxCacheBreakpoints is the number of cache_control markers the API
// accepts on one request.
const maxCacheBreakpoints = 4

// cacheBreakpoint remembers which block a cache hint must land on once the
// turns are final: the block is identified by the pointer it wraps, which
// survives the reordering done by orderTurn.
type cacheBreakpoint struct {
	turn  int
	block anthropicsdk.ContentBlockParamUnion
	cc    *anthropicsdk.CacheControlEphemeralParam
}

// sameBlock reports whether two unions wrap the same underlying block.
func sameBlock(a, b anthropicsdk.ContentBlockParamUnion) bool {
	return a.OfText == b.OfText && a.OfImage == b.OfImage && a.OfDocument == b.OfDocument &&
		a.OfToolUse == b.OfToolUse && a.OfToolResult == b.OfToolResult &&
		a.OfThinking == b.OfThinking && a.OfRedactedThinking == b.OfRedactedThinking
}

// orderTurn restores the block order the API expects inside a turn that
// was assembled from several messages: thinking blocks lead an assistant
// turn, tool results lead a user turn. The relative order within each group
// is preserved.
func orderTurn(role anthropicsdk.MessageParamRole, blocks []anthropicsdk.ContentBlockParamUnion) []anthropicsdk.ContentBlockParamUnion {
	leads := func(b anthropicsdk.ContentBlockParamUnion) bool {
		switch role {
		case anthropicsdk.MessageParamRoleAssistant:
			return b.OfThinking != nil || b.OfRedactedThinking != nil
		case anthropicsdk.MessageParamRoleUser:
			return b.OfToolResult != nil
		}
		return false
	}

	ordered := make([]anthropicsdk.ContentBlockParamUnion, 0, len(blocks))
	for _, b := range blocks {
		if leads(b) {
			ordered = append(ordered, b)
		}
	}
	for _, b := range blocks {
		if !leads(b) {
			ordered = append(ordered, b)
		}
	}
	return ordered
}

// userBlocks builds the content of a user turn: its text first, then one
// block per attachment.
func userBlocks(m llm.Message) ([]anthropicsdk.ContentBlockParamUnion, error) {
	blocks := make([]anthropicsdk.ContentBlockParamUnion, 0, 1+len(m.Attachments()))
	if content := m.Content(); content != "" {
		blocks = append(blocks, anthropicsdk.NewTextBlock(content))
	}
	for _, attachment := range m.Attachments() {
		block, err := attachmentBlock(attachment)
		if err != nil {
			return nil, errors.WithStack(err)
		}
		blocks = append(blocks, block)
	}
	return blocks, nil
}

// reasoningBlocks replays the thinking blocks of a previous assistant turn,
// which must precede its text and tool_use blocks. Only blocks the API can
// verify are replayed: signed thinking and redacted (encrypted) thinking.
// A bare reasoning string has no signature and would be rejected, so it is
// dropped.
func reasoningBlocks(m llm.Message) []anthropicsdk.ContentBlockParamUnion {
	rm, ok := m.(llm.ReasoningMessage)
	if !ok {
		return nil
	}
	var blocks []anthropicsdk.ContentBlockParamUnion
	for _, d := range rm.ReasoningDetails() {
		switch d.Type {
		case llm.ReasoningDetailTypeEncrypted:
			if d.Data != "" {
				blocks = append(blocks, anthropicsdk.NewRedactedThinkingBlock(d.Data))
			}
		case llm.ReasoningDetailTypeText:
			if d.Signature != "" {
				blocks = append(blocks, anthropicsdk.NewThinkingBlock(d.Signature, d.Text))
			}
		}
	}
	return blocks
}

// hasReasoning reports whether an assistant message carries any reasoning,
// replayable or not.
func hasReasoning(m llm.Message) bool {
	rm, ok := m.(llm.ReasoningMessage)
	return ok && (rm.Reasoning() != "" || len(rm.ReasoningDetails()) > 0)
}

// toolCallInput decodes the tool call parameters, kept as a JSON string by
// genai, into the object the tool_use block carries.
func toolCallInput(parameters any) (any, error) {
	var raw []byte
	switch typed := parameters.(type) {
	case string:
		raw = []byte(typed)
	case []byte:
		raw = typed
	case nil:
		return map[string]any{}, nil
	default:
		return typed, nil
	}
	if len(raw) == 0 {
		return map[string]any{}, nil
	}
	var input any
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, errors.WithStack(err)
	}
	if input == nil {
		return map[string]any{}, nil
	}
	return input, nil
}

// cacheControl converts the cache hint carried by a message, if any. The
// API only knows the ephemeral type with a 5m or 1h TTL; anything else is
// refused here rather than by a remote 400.
func cacheControl(m llm.Message) (*anthropicsdk.CacheControlEphemeralParam, error) {
	cm, ok := m.(llm.CacheControlMessage)
	if !ok {
		return nil, nil
	}
	cc := cm.CacheControl()
	if cc == nil {
		return nil, nil
	}
	if cc.Type != "" && cc.Type != "ephemeral" {
		return nil, llm.NewValidationError("cache_control", fmt.Sprintf("unsupported cache control type %q (only ephemeral)", cc.Type))
	}
	param := anthropicsdk.NewCacheControlEphemeralParam()
	if cc.TTL != nil && *cc.TTL != "" {
		switch ttl := anthropicsdk.CacheControlEphemeralTTL(*cc.TTL); ttl {
		case anthropicsdk.CacheControlEphemeralTTLTTL5m, anthropicsdk.CacheControlEphemeralTTLTTL1h:
			param.TTL = ttl
		default:
			return nil, llm.NewValidationError("cache_control", fmt.Sprintf("unsupported cache TTL %q (only 5m or 1h)", *cc.TTL))
		}
	}
	return &param, nil
}

// applyCacheControl sets the cache breakpoint on the last block of a turn,
// which is where the API expects it: everything up to and including that
// block becomes the cached prefix. Thinking blocks cannot carry one; a turn
// made of thinking only (an assistant message with no text) therefore
// drops the hint, which is harmless: such a turn is never the end of a
// prefix worth caching on its own.
func applyCacheControl(blocks []anthropicsdk.ContentBlockParamUnion, cc *anthropicsdk.CacheControlEphemeralParam) {
	if cc == nil || len(blocks) == 0 {
		return
	}
	last := &blocks[len(blocks)-1]
	switch {
	case last.OfText != nil:
		last.OfText.CacheControl = *cc
	case last.OfImage != nil:
		last.OfImage.CacheControl = *cc
	case last.OfDocument != nil:
		last.OfDocument.CacheControl = *cc
	case last.OfToolUse != nil:
		last.OfToolUse.CacheControl = *cc
	case last.OfToolResult != nil:
		last.OfToolResult.CacheControl = *cc
	}
}

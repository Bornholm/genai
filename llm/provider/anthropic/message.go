package anthropic

import (
	"encoding/json"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// buildMessages converts genai messages to the Messages API shape: system
// prompts are hoisted into the dedicated system field, and the rest is
// folded into strictly alternating user/assistant turns.
//
// Folding matters beyond cosmetics: a tool_result must live in the user
// turn that immediately follows the assistant turn carrying its tool_use,
// so consecutive tool messages have to share one user turn.
func buildMessages(msgs []llm.Message) ([]anthropicsdk.TextBlockParam, []anthropicsdk.MessageParam, error) {
	var (
		system   []anthropicsdk.TextBlockParam
		messages []anthropicsdk.MessageParam
	)

	add := func(role anthropicsdk.MessageParamRole, blocks []anthropicsdk.ContentBlockParamUnion) {
		if len(blocks) == 0 {
			return
		}
		if n := len(messages); n > 0 && messages[n-1].Role == role {
			messages[n-1].Content = append(messages[n-1].Content, blocks...)
			return
		}
		messages = append(messages, anthropicsdk.MessageParam{Role: role, Content: blocks})
	}

	for _, m := range msgs {
		cc := cacheControl(m)

		switch m.Role() {
		case llm.RoleSystem:
			if len(m.Attachments()) > 0 {
				return nil, nil, errors.New("system messages cannot have attachments")
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
			applyCacheControl(blocks, cc)
			add(anthropicsdk.MessageParamRoleUser, blocks)

		case llm.RoleAssistant:
			if len(m.Attachments()) > 0 {
				return nil, nil, errors.New("assistant messages cannot have attachments")
			}
			blocks := reasoningBlocks(m)
			if content := m.Content(); content != "" {
				blocks = append(blocks, anthropicsdk.NewTextBlock(content))
			}
			applyCacheControl(blocks, cc)
			add(anthropicsdk.MessageParamRoleAssistant, blocks)

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
			applyCacheControl(blocks, cc)
			add(anthropicsdk.MessageParamRoleAssistant, blocks)

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
			if cc != nil {
				result.CacheControl = *cc
			}
			add(anthropicsdk.MessageParamRoleUser, []anthropicsdk.ContentBlockParamUnion{{OfToolResult: &result}})

		default:
			return nil, nil, errors.Errorf("unsupported message role '%s'", m.Role())
		}
	}

	return system, messages, nil
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

// cacheControl converts the cache hint carried by a message, if any.
func cacheControl(m llm.Message) *anthropicsdk.CacheControlEphemeralParam {
	cm, ok := m.(llm.CacheControlMessage)
	if !ok {
		return nil
	}
	cc := cm.CacheControl()
	if cc == nil {
		return nil
	}
	param := anthropicsdk.NewCacheControlEphemeralParam()
	if cc.TTL != nil && *cc.TTL != "" {
		param.TTL = anthropicsdk.CacheControlEphemeralTTL(*cc.TTL)
	}
	return &param
}

// applyCacheControl sets the cache breakpoint on the last block of a turn,
// which is where the API expects it: everything up to and including that
// block becomes the cached prefix.
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

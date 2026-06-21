package proxy

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/google/uuid"
	"github.com/pkg/errors"
)

// ---- Anthropic Messages wire types --------------------------------------

// anthropicMessagesRequest mirrors the Anthropic /v1/messages request body.
type anthropicMessagesRequest struct {
	Model         string                   `json:"model"`
	MaxTokens     int                      `json:"max_tokens"`
	Messages      []anthropicMessage       `json:"messages"`
	System        any                      `json:"system,omitempty"` // string or []content block
	Tools         []anthropicTool          `json:"tools,omitempty"`
	ToolChoice    json.RawMessage          `json:"tool_choice,omitempty"`
	Temperature   *float64                 `json:"temperature,omitempty"`
	TopP          *float64                 `json:"top_p,omitempty"`
	TopK          *int                     `json:"top_k,omitempty"`
	StopSequences []string                 `json:"stop_sequences,omitempty"`
	Stream        bool                     `json:"stream,omitempty"`
	Thinking      *anthropicThinkingConfig `json:"thinking,omitempty"`
}

// anthropicThinkingConfig configures Anthropic's "extended thinking" feature.
type anthropicThinkingConfig struct {
	Type         string `json:"type"` // "enabled" | "disabled" | "adaptive"
	BudgetTokens *int   `json:"budget_tokens,omitempty"`
}

// anthropicMessage is a single turn in the conversation. Content is either a
// plain string or an array of content blocks (decoded as []any / map[string]any).
type anthropicMessage struct {
	Role    string `json:"role"` // "user" | "assistant"
	Content any    `json:"content"`
}

// anthropicTool mirrors an Anthropic tool definition.
type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
}

// anthropicUsage mirrors the "usage" object of an Anthropic Messages response.
type anthropicUsage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens,omitempty"`
}

// anthropicMessagesResponse mirrors the Anthropic /v1/messages response body.
// Content blocks are encoded as plain maps so each block type only carries
// the fields it actually needs.
type anthropicMessagesResponse struct {
	ID           string           `json:"id"`
	Type         string           `json:"type"`
	Role         string           `json:"role"`
	Model        string           `json:"model"`
	Content      []map[string]any `json:"content"`
	StopReason   string           `json:"stop_reason"`
	StopSequence *string          `json:"stop_sequence"`
	Usage        anthropicUsage   `json:"usage"`
}

// ---- Request conversion ---------------------------------------------------

// ParseMessagesRequest converts an Anthropic Messages JSON body to llm options.
func ParseMessagesRequest(body json.RawMessage) (model string, stream bool, opts []llm.ChatCompletionOptionFunc, err error) {
	var req anthropicMessagesRequest
	if err = json.Unmarshal(body, &req); err != nil {
		return "", false, nil, errors.Wrap(err, "could not parse messages request")
	}

	model = req.Model
	stream = req.Stream

	messages, err := convertAnthropicMessages(req.System, req.Messages)
	if err != nil {
		return "", false, nil, errors.Wrap(err, "could not convert messages")
	}
	opts = append(opts, llm.WithMessages(messages...))

	if req.MaxTokens > 0 {
		opts = append(opts, llm.WithMaxCompletionTokens(req.MaxTokens))
	}

	if req.Temperature != nil {
		opts = append(opts, llm.WithTemperature(*req.Temperature))
	}

	if len(req.StopSequences) > 0 || req.TopP != nil || req.TopK != nil {
		slog.Debug("ignoring unsupported anthropic request fields",
			slog.Any("stop_sequences", req.StopSequences),
			slog.Any("top_p", req.TopP),
			slog.Any("top_k", req.TopK),
		)
	}

	if len(req.Tools) > 0 {
		tools := make([]llm.Tool, 0, len(req.Tools))
		for _, t := range req.Tools {
			schema := t.InputSchema
			if schema == nil {
				schema = map[string]any{}
			}
			tools = append(tools, llm.NewFuncTool(t.Name, t.Description, schema, nil))
		}
		opts = append(opts, llm.WithTools(tools...))
	}

	if len(req.ToolChoice) > 0 {
		var toolChoice map[string]any
		if jsonErr := json.Unmarshal(req.ToolChoice, &toolChoice); jsonErr == nil {
			switch toolChoice["type"] {
			case "auto":
				opts = append(opts, llm.WithToolChoice(llm.ToolChoiceAuto))
			case "any":
				opts = append(opts, llm.WithToolChoice(llm.ToolChoiceRequired))
			case "none":
				opts = append(opts, llm.WithToolChoice(llm.ToolChoiceNone))
			case "tool":
				slog.Debug("anthropic tool_choice forcing a specific tool is not supported, falling back to required",
					slog.Any("name", toolChoice["name"]),
				)
				opts = append(opts, llm.WithToolChoice(llm.ToolChoiceRequired))
			}
		}
	}

	if req.Thinking != nil && req.Thinking.Type == "enabled" && req.Thinking.BudgetTokens != nil {
		budget := *req.Thinking.BudgetTokens
		enabled := true
		opts = append(opts, llm.WithReasoning(&llm.ReasoningOptions{
			MaxTokens: &budget,
			Enabled:   &enabled,
		}))
	}

	return model, stream, opts, nil
}

// ConvertAnthropicMessagesJSON converts a JSON array of Anthropic Messages
// API turns into genai's internal []llm.Message representation. The "system"
// prompt (if any) is expected to be carried separately and is not handled
// here.
func ConvertAnthropicMessagesJSON(messagesJSON json.RawMessage) ([]llm.Message, error) {
	var messages []anthropicMessage
	if err := json.Unmarshal(messagesJSON, &messages); err != nil {
		return nil, errors.Wrap(err, "could not unmarshal messages")
	}

	return convertAnthropicMessages(nil, messages)
}

// convertAnthropicMessages converts the system prompt and conversation
// messages of an Anthropic Messages request into llm.Message values.
func convertAnthropicMessages(system any, messages []anthropicMessage) ([]llm.Message, error) {
	out := make([]llm.Message, 0, len(messages)+1)

	systemMessages, err := convertAnthropicSystem(system)
	if err != nil {
		return nil, err
	}
	out = append(out, systemMessages...)

	for _, m := range messages {
		switch m.Role {
		case "user":
			msgs, err := convertAnthropicUserMessage(m.Content)
			if err != nil {
				return nil, err
			}
			out = append(out, msgs...)
		case "assistant":
			msgs, err := convertAnthropicAssistantMessage(m.Content)
			if err != nil {
				return nil, err
			}
			out = append(out, msgs...)
		default:
			text, attachments, cacheControl, err := extractAnthropicContentParts(m.Content)
			if err != nil {
				return nil, errors.Wrapf(err, "could not convert content for role %s", m.Role)
			}
			out = append(out, newMessageWithParts(llm.Role(m.Role), text, attachments, cacheControl))
		}
	}

	return out, nil
}

// convertAnthropicSystem converts the "system" field (string or array of text
// blocks with optional cache_control) into one or more system messages.
func convertAnthropicSystem(system any) ([]llm.Message, error) {
	if system == nil {
		return nil, nil
	}

	switch v := system.(type) {
	case string:
		if v == "" {
			return nil, nil
		}
		return []llm.Message{llm.NewMessage(llm.RoleSystem, v)}, nil
	case []any:
		out := make([]llm.Message, 0, len(v))
		for _, item := range v {
			block, ok := item.(map[string]any)
			if !ok {
				continue
			}
			text, _ := block["text"].(string)
			if text == "" {
				continue
			}
			if cc := extractCacheControl(block["cache_control"]); cc != nil {
				out = append(out, llm.NewMessageWithCacheControl(llm.RoleSystem, text, cc))
			} else {
				out = append(out, llm.NewMessage(llm.RoleSystem, text))
			}
		}
		return out, nil
	default:
		return nil, errors.Errorf("unsupported system prompt type %T", system)
	}
}

// convertAnthropicUserMessage converts a "user" message's content into one or
// more llm.Message values. Each tool_result block becomes its own
// llm.ToolMessage; any remaining text/image/document blocks are combined into
// a single user message.
func convertAnthropicUserMessage(content any) ([]llm.Message, error) {
	blocks, ok := content.([]any)
	if !ok {
		text, attachments, cacheControl, err := extractAnthropicContentParts(content)
		if err != nil {
			return nil, errors.Wrap(err, "could not convert user message content")
		}
		if text == "" && len(attachments) == 0 {
			return nil, nil
		}
		return []llm.Message{newMessageWithParts(llm.RoleUser, text, attachments, cacheControl)}, nil
	}

	var out []llm.Message
	var otherBlocks []any

	for _, item := range blocks {
		block, ok := item.(map[string]any)
		if !ok {
			continue
		}

		if block["type"] != "tool_result" {
			otherBlocks = append(otherBlocks, item)
			continue
		}

		toolUseID, _ := block["tool_use_id"].(string)
		text, attachments, _, err := extractAnthropicContentParts(block["content"])
		if err != nil {
			return nil, errors.Wrap(err, "could not convert tool_result content")
		}
		if isError, _ := block["is_error"].(bool); isError {
			text = "Error: " + text
		}
		out = append(out, llm.NewToolMessage(toolUseID, llm.NewToolResult(text, attachments...)))
	}

	if len(otherBlocks) > 0 {
		text, attachments, cacheControl, err := extractAnthropicContentParts(otherBlocks)
		if err != nil {
			return nil, errors.Wrap(err, "could not convert user message content")
		}
		if text != "" || len(attachments) > 0 {
			out = append(out, newMessageWithParts(llm.RoleUser, text, attachments, cacheControl))
		}
	}

	return out, nil
}

// convertAnthropicAssistantMessage converts an "assistant" message's content
// (which may mix text, thinking and tool_use blocks) into one or two
// llm.Message values: an optional text/reasoning message followed by an
// optional tool calls message.
func convertAnthropicAssistantMessage(content any) ([]llm.Message, error) {
	blocks, ok := content.([]any)
	if !ok {
		text, _, _, err := extractAnthropicContentParts(content)
		if err != nil {
			return nil, errors.Wrap(err, "could not convert assistant message content")
		}
		if text == "" {
			return nil, nil
		}
		return []llm.Message{llm.NewMessage(llm.RoleAssistant, text)}, nil
	}

	var (
		textBuf   strings.Builder
		reasoning strings.Builder
		details   []llm.ReasoningDetail
		toolCalls []llm.ToolCall
	)

	for _, item := range blocks {
		block, ok := item.(map[string]any)
		if !ok {
			continue
		}

		switch block["type"] {
		case "text":
			if t, ok := block["text"].(string); ok {
				textBuf.WriteString(t)
			}
		case "thinking":
			thinking, _ := block["thinking"].(string)
			signature, _ := block["signature"].(string)
			reasoning.WriteString(thinking)
			details = append(details, llm.ReasoningDetail{
				Type:      llm.ReasoningDetailTypeText,
				Text:      thinking,
				Signature: signature,
				Index:     len(details),
			})
		case "redacted_thinking":
			data, _ := block["data"].(string)
			details = append(details, llm.ReasoningDetail{
				Type:  llm.ReasoningDetailTypeEncrypted,
				Data:  data,
				Index: len(details),
			})
		case "tool_use":
			id, _ := block["id"].(string)
			name, _ := block["name"].(string)
			input := block["input"]
			if input == nil {
				input = map[string]any{}
			}
			raw, err := json.Marshal(input)
			if err != nil {
				return nil, errors.Wrap(err, "could not marshal tool_use input")
			}
			toolCalls = append(toolCalls, llm.NewToolCall(id, name, string(raw)))
		}
	}

	text := textBuf.String()
	reasoningText := reasoning.String()
	hasReasoning := reasoningText != "" || len(details) > 0

	var out []llm.Message

	if len(toolCalls) == 0 {
		switch {
		case hasReasoning:
			out = append(out, llm.NewAssistantReasoningMessage(text, reasoningText, details))
		case text != "":
			out = append(out, llm.NewMessage(llm.RoleAssistant, text))
		}
		return out, nil
	}

	if text != "" {
		out = append(out, llm.NewMessage(llm.RoleAssistant, text))
	}
	if hasReasoning {
		out = append(out, llm.NewReasoningToolCallsMessage(reasoningText, details, toolCalls...))
	} else {
		out = append(out, llm.NewToolCallsMessage(toolCalls...))
	}

	return out, nil
}

// newMessageWithParts builds the simplest llm.Message variant that fits the
// given text/attachments/cache control combination.
func newMessageWithParts(role llm.Role, text string, attachments []llm.Attachment, cacheControl *llm.CacheControl) llm.Message {
	if len(attachments) > 0 {
		return llm.NewMultimodalMessage(role, text, attachments...)
	}
	if cacheControl != nil {
		return llm.NewMessageWithCacheControl(role, text, cacheControl)
	}
	return llm.NewMessage(role, text)
}

// extractAnthropicContentParts extracts text, attachments and an optional
// cache control hint from an Anthropic content field (string or array of
// content blocks). It handles "text", "image" and "document" blocks; other
// block types (e.g. "tool_use", "tool_result") are ignored by this function
// and must be handled by the caller.
func extractAnthropicContentParts(raw any) (text string, attachments []llm.Attachment, cacheControl *llm.CacheControl, err error) {
	if raw == nil {
		return "", nil, nil, nil
	}

	switch v := raw.(type) {
	case string:
		return v, nil, nil, nil
	case []any:
		var buf strings.Builder
		for _, item := range v {
			block, ok := item.(map[string]any)
			if !ok {
				continue
			}

			switch block["type"] {
			case "text":
				if t, ok := block["text"].(string); ok {
					buf.WriteString(t)
				}
			case "image":
				source, ok := block["source"].(map[string]any)
				if !ok {
					continue
				}
				att, attErr := convertAnthropicSource(llm.AttachmentTypeImage, source)
				if attErr != nil {
					return "", nil, nil, errors.Wrap(attErr, "could not convert image block")
				}
				attachments = append(attachments, att)
			case "document":
				source, ok := block["source"].(map[string]any)
				if !ok {
					continue
				}
				switch source["type"] {
				case "base64", "url":
					att, attErr := convertAnthropicSource(llm.AttachmentTypeDocument, source)
					if attErr != nil {
						return "", nil, nil, errors.Wrap(attErr, "could not convert document block")
					}
					attachments = append(attachments, att)
				case "text":
					// Plain-text document source — inline its content as text.
					// "content" (PDF made of nested blocks) is not supported.
					if t, ok := source["data"].(string); ok {
						buf.WriteString(t)
					}
				}
			}

			if cc := extractCacheControl(block["cache_control"]); cc != nil {
				cacheControl = cc
			}
		}
		return buf.String(), attachments, cacheControl, nil
	default:
		return fmt.Sprintf("%v", raw), nil, nil, nil
	}
}

// convertAnthropicSource creates an llm.Attachment from an Anthropic content
// block "source" object (base64 or url).
func convertAnthropicSource(attachmentType llm.AttachmentType, source map[string]any) (llm.Attachment, error) {
	switch source["type"] {
	case "base64":
		mediaType, _ := source["media_type"].(string)
		data, _ := source["data"].(string)
		if data == "" {
			return nil, errors.New("missing base64 data")
		}
		return llm.NewBase64Attachment(attachmentType, mediaType, data)
	case "url":
		url, _ := source["url"].(string)
		if url == "" {
			return nil, errors.New("missing source url")
		}
		mediaType, _ := source["media_type"].(string)
		if mediaType == "" {
			if attachmentType == llm.AttachmentTypeDocument {
				mediaType = "application/octet-stream"
			} else {
				mediaType = "image/*"
			}
		}
		return llm.NewURLAttachment(attachmentType, mediaType, url)
	default:
		return nil, errors.Errorf("unsupported source type %v", source["type"])
	}
}

// ---- Response conversion ---------------------------------------------------

// FormatMessagesResponse converts a llm.ChatCompletionResponse to an
// Anthropic Messages JSON response.
func FormatMessagesResponse(res llm.ChatCompletionResponse, model string) any {
	var blocks []map[string]any

	if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok {
		details := rr.ReasoningDetails()
		switch {
		case len(details) > 0:
			for _, d := range details {
				if d.Type == llm.ReasoningDetailTypeEncrypted {
					blocks = append(blocks, map[string]any{
						"type": "redacted_thinking",
						"data": d.Data,
					})
					continue
				}
				block := map[string]any{
					"type":     "thinking",
					"thinking": d.Text,
				}
				if d.Signature != "" {
					block["signature"] = d.Signature
				}
				blocks = append(blocks, block)
			}
		case rr.Reasoning() != "":
			blocks = append(blocks, map[string]any{
				"type":     "thinking",
				"thinking": rr.Reasoning(),
			})
		}
	}

	if content := res.Message().Content(); content != "" {
		blocks = append(blocks, map[string]any{
			"type": "text",
			"text": content,
		})
	}

	stopReason := "end_turn"

	if toolCalls := res.ToolCalls(); len(toolCalls) > 0 {
		stopReason = "tool_use"
		for _, tc := range toolCalls {
			blocks = append(blocks, map[string]any{
				"type":  "tool_use",
				"id":    tc.ID(),
				"name":  tc.Name(),
				"input": parseToolCallInput(tc.Parameters()),
			})
		}
	}

	if blocks == nil {
		blocks = []map[string]any{}
	}

	usage := res.Usage()
	out := anthropicUsage{
		InputTokens:  usage.PromptTokens(),
		OutputTokens: usage.CompletionTokens(),
	}
	type cachedUsage interface{ CachedTokens() int64 }
	if cu, ok := usage.(cachedUsage); ok {
		out.CacheReadInputTokens = cu.CachedTokens()
	}

	return anthropicMessagesResponse{
		ID:         "msg_" + uuid.New().String(),
		Type:       "message",
		Role:       "assistant",
		Model:      model,
		Content:    blocks,
		StopReason: stopReason,
		Usage:      out,
	}
}

// parseToolCallInput decodes a llm.ToolCall's Parameters() (typically a JSON
// string) into a JSON object suitable for the "input" field of a tool_use block.
func parseToolCallInput(params any) any {
	switch p := params.(type) {
	case string:
		if p == "" {
			return map[string]any{}
		}
		var input any
		if err := json.Unmarshal([]byte(p), &input); err != nil {
			return map[string]any{}
		}
		return input
	case nil:
		return map[string]any{}
	default:
		return p
	}
}

// ---- Token count estimation -------------------------------------------------

// estimatedCharsPerToken is a rough Claude-tokenizer approximation: there is
// no Anthropic tokenizer available in Go, and using an OpenAI tokenizer
// (different vocabulary/BPE) would give a false sense of precision. This is
// only meant to be in the right order of magnitude for context-window
// management — it must not be used for billing.
const estimatedCharsPerToken = 4

// estimatedTokensPerImage is a flat per-image token estimate, used because
// the proxy does not decode image dimensions.
const estimatedTokensPerImage = 1600

// EstimateTokenCount returns an approximate input token count for an
// Anthropic Messages request body. This is a heuristic (~chars/4, plus a flat
// per-image estimate) and not an exact count.
func EstimateTokenCount(body json.RawMessage) int {
	var req anthropicMessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return len(body) / estimatedCharsPerToken
	}

	var totalChars int
	var imageCount int

	switch v := req.System.(type) {
	case string:
		totalChars += len(v)
	case []any:
		if raw, err := json.Marshal(v); err == nil {
			totalChars += len(raw)
		}
	}

	for _, m := range req.Messages {
		switch v := m.Content.(type) {
		case string:
			totalChars += len(v)
		case []any:
			for _, item := range v {
				block, ok := item.(map[string]any)
				if !ok {
					continue
				}
				if block["type"] == "image" {
					imageCount++
					continue
				}
				if raw, err := json.Marshal(block); err == nil {
					totalChars += len(raw)
				}
			}
		}
	}

	for _, t := range req.Tools {
		totalChars += len(t.Name) + len(t.Description)
		if raw, err := json.Marshal(t.InputSchema); err == nil {
			totalChars += len(raw)
		}
	}

	return totalChars/estimatedCharsPerToken + imageCount*estimatedTokensPerImage
}

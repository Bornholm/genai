package proxy

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"mime"
	neturl "net/url"
	"path"
	"strings"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/google/uuid"
	"github.com/pkg/errors"
)

// ---- OpenAI wire types --------------------------------------------------

// openAIChatRequest mirrors the OpenAI /v1/chat/completions request body.
type openAIChatRequest struct {
	Model           string             `json:"model"`
	Messages        []openAIMessage    `json:"messages"`
	Tools           []openAITool       `json:"tools,omitempty"`
	ToolChoice      any                `json:"tool_choice,omitempty"`
	Temperature     *float64           `json:"temperature,omitempty"`
	MaxTokens       *int               `json:"max_tokens,omitempty"`
	Stream          bool               `json:"stream"`
	Seed            *int               `json:"seed,omitempty"`
	ResponseFmt     *openAIResponseFmt `json:"response_format,omitempty"`
	ReasoningEffort string             `json:"reasoning_effort,omitempty"` // e.g. "low","medium","high"
}

// handledRequestFields lists the request keys ParseChatCompletionRequest maps to
// explicit llm options, plus the ones the provider layer owns. Everything else
// is forwarded verbatim (see passthroughRequestFields), so that parameters this
// struct does not model — parallel_tool_calls, top_p, stop, frequency_penalty,
// presence_penalty, user… — are not silently dropped on the way to the provider.
var handledRequestFields = map[string]struct{}{
	"model":            {},
	"messages":         {},
	"tools":            {},
	"temperature":      {},
	"max_tokens":       {},
	"stream":           {},
	"seed":             {},
	"response_format":  {},
	"reasoning_effort": {},

	// Owned by the provider layer: the OpenAI SDK sets stream_options itself
	// (include_usage) and overriding it here would break usage tracking.
	"stream_options": {},
	// The proxy exposes a single choice; forwarding n>1 would produce a
	// response the response formatters cannot represent.
	"n": {},
}

// passthroughRequestFields returns the request body fields that have no
// explicit mapping, to be forwarded verbatim as provider extra fields.
//
// tool_choice is a special case: the string forms are mapped to llm.ToolChoice
// and dropped here, but the object form ({"type":"function",...}) has no llm
// equivalent, so it is passed through to preserve the constraint.
func passthroughRequestFields(body json.RawMessage) map[string]any {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil
	}

	for k := range handledRequestFields {
		delete(raw, k)
	}
	if _, isString := raw["tool_choice"].(string); isString {
		delete(raw, "tool_choice")
	}

	if len(raw) == 0 {
		return nil
	}
	return raw
}

type openAIResponseFmt struct {
	Type string `json:"type"` // "text" | "json_object" | "json_schema"
}

type openAIMessage struct {
	Role             string           `json:"role"`
	Content          any              `json:"content"` // string or array of content parts
	ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	Name             string           `json:"name,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"` // reasoning tokens in response / multi-turn
	// ReasoningDetails is the structured form of the reasoning content, using
	// the OpenRouter convention. Clients built on the Vercel AI SDK replay
	// reasoning through this field rather than reasoning_content; dropping it
	// loses the model's reasoning across turns.
	ReasoningDetails []openAIReasoningDetail `json:"reasoning_details,omitempty"`
}

// openAIReasoningDetail mirrors OpenRouter's reasoning_details block, the de
// facto standard for carrying structured reasoning over the OpenAI wire format.
type openAIReasoningDetail struct {
	ID        string `json:"id,omitempty"`
	Type      string `json:"type,omitempty"`
	Text      string `json:"text,omitempty"`
	Summary   string `json:"summary,omitempty"`
	Data      string `json:"data,omitempty"`
	Format    string `json:"format,omitempty"`
	Index     int    `json:"index,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// toLLMReasoningDetails converts wire reasoning details to their llm form.
func toLLMReasoningDetails(details []openAIReasoningDetail) []llm.ReasoningDetail {
	if len(details) == 0 {
		return nil
	}
	out := make([]llm.ReasoningDetail, 0, len(details))
	for _, d := range details {
		out = append(out, llm.ReasoningDetail{
			ID:        d.ID,
			Type:      llm.ReasoningDetailType(d.Type),
			Text:      d.Text,
			Summary:   d.Summary,
			Data:      d.Data,
			Format:    d.Format,
			Index:     d.Index,
			Signature: d.Signature,
		})
	}
	return out
}

// fromLLMReasoningDetails converts llm reasoning details to their wire form.
func fromLLMReasoningDetails(details []llm.ReasoningDetail) []openAIReasoningDetail {
	if len(details) == 0 {
		return nil
	}
	out := make([]openAIReasoningDetail, 0, len(details))
	for _, d := range details {
		out = append(out, openAIReasoningDetail{
			ID:        d.ID,
			Type:      string(d.Type),
			Text:      d.Text,
			Summary:   d.Summary,
			Data:      d.Data,
			Format:    d.Format,
			Index:     d.Index,
			Signature: d.Signature,
		})
	}
	return out
}

type openAITool struct {
	Type     string         `json:"type"` // "function"
	Function openAIFunction `json:"function"`
}

type openAIFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type openAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"` // "function"
	Function openAIFunctionCall `json:"function"`
}

type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ---- OpenAI response types ----------------------------------------------

type openAIChatResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   openAIUsage    `json:"usage"`
}

type openAIChoice struct {
	Index        int           `json:"index"`
	Message      openAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// openAIStreamChunk is a single SSE data payload.
type openAIStreamChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []openAIStreamChoice `json:"choices"`
	Usage   *openAIUsage         `json:"usage,omitempty"`
}

type openAIStreamChoice struct {
	Index        int               `json:"index"`
	Delta        openAIStreamDelta `json:"delta"`
	FinishReason *string           `json:"finish_reason"`
}

type openAIStreamDelta struct {
	Role             string                 `json:"role,omitempty"`
	Content          string                 `json:"content,omitempty"`
	ToolCalls        []openAIStreamToolCall `json:"tool_calls,omitempty"`
	ReasoningContent string                 `json:"reasoning_content,omitempty"`

	ReasoningDetails []openAIReasoningDetail `json:"reasoning_details,omitempty"`
}

// openAIStreamToolCall is the tool call format used inside streaming deltas.
// It includes an Index field required by OpenAI-compatible clients.
type openAIStreamToolCall struct {
	Index    int                `json:"index"`
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function openAIFunctionCall `json:"function"`
}

// ---- Embeddings wire types ----------------------------------------------

type openAIEmbeddingRequest struct {
	Model      string `json:"model"`
	Input      any    `json:"input"` // string or []string
	Dimensions *int   `json:"dimensions,omitempty"`
}

type openAIEmbeddingResponse struct {
	Object string               `json:"object"`
	Model  string               `json:"model"`
	Data   []openAIEmbeddingObj `json:"data"`
	Usage  openAIEmbeddingUsage `json:"usage"`
}

type openAIEmbeddingObj struct {
	Index     int       `json:"index"`
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
}

type openAIEmbeddingUsage struct {
	PromptTokens int64 `json:"prompt_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}

// ---- Models wire type ---------------------------------------------------

type openAIModelsResponse struct {
	Object string           `json:"object"`
	Data   []openAIModelObj `json:"data"`
}

type openAIModelObj struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ---- Conversion helpers -------------------------------------------------

// ParseChatCompletionRequest converts an OpenAI JSON body to llm options.
func ParseChatCompletionRequest(body json.RawMessage) (model string, stream bool, opts []llm.ChatCompletionOptionFunc, err error) {
	var req openAIChatRequest
	if err = json.Unmarshal(body, &req); err != nil {
		return "", false, nil, errors.Wrap(err, "could not parse chat completion request")
	}

	model = req.Model
	stream = req.Stream

	messages, err := convertMessages(req.Messages)
	if err != nil {
		return "", false, nil, errors.Wrap(err, "could not convert messages")
	}
	opts = append(opts, llm.WithMessages(messages...))

	if req.Temperature != nil {
		opts = append(opts, llm.WithTemperature(*req.Temperature))
	}
	if req.MaxTokens != nil {
		opts = append(opts, llm.WithMaxCompletionTokens(*req.MaxTokens))
	}
	if req.Seed != nil {
		opts = append(opts, llm.WithSeed(*req.Seed))
	}

	if req.ReasoningEffort != "" {
		effort := llm.ReasoningEffort(req.ReasoningEffort)
		opts = append(opts, llm.WithReasoning(llm.NewReasoningOptions(effort)))
	}

	if req.ResponseFmt != nil {
		switch req.ResponseFmt.Type {
		case "json_object":
			opts = append(opts, llm.WithResponseFormat(llm.ResponseFormatJSON))
		}
	}

	if len(req.Tools) > 0 {
		tools := make([]llm.Tool, 0, len(req.Tools))
		for _, t := range req.Tools {
			params := t.Function.Parameters
			if params == nil {
				params = map[string]any{}
			}
			paramsMap, ok := params.(map[string]any)
			if !ok {
				// re-marshal/unmarshal to get map
				raw, err2 := json.Marshal(params)
				if err2 == nil {
					_ = json.Unmarshal(raw, &paramsMap)
				}
			}
			tools = append(tools, llm.NewFuncTool(
				t.Function.Name,
				t.Function.Description,
				paramsMap,
				nil, // proxy doesn't execute tools
			))
		}
		opts = append(opts, llm.WithTools(tools...))
	}

	// Tool choice
	if req.ToolChoice != nil {
		switch v := req.ToolChoice.(type) {
		case string:
			switch v {
			case "none":
				opts = append(opts, llm.WithToolChoice(llm.ToolChoiceNone))
			case "auto":
				opts = append(opts, llm.WithToolChoice(llm.ToolChoiceAuto))
			case "required":
				opts = append(opts, llm.WithToolChoice(llm.ToolChoiceRequired))
			}
		default:
			// Object form: {"type":"function","function":{"name":"..."}}.
			// llm.ToolChoice cannot name a function, so require a call here and
			// let the verbatim tool_choice pass-through carry the exact
			// constraint to the provider.
			opts = append(opts, llm.WithToolChoice(llm.ToolChoiceRequired))
		}
	} else if len(req.Tools) > 0 {
		// OpenAI's API defaults to "auto" when tools are provided without an
		// explicit tool_choice. llm.NewChatCompletionOptions defaults to
		// ToolChoiceNone, which would silently prevent the model from ever
		// calling a tool.
		opts = append(opts, llm.WithToolChoice(llm.ToolChoiceAuto))
	}

	// Forward every parameter without an explicit mapping. Appended last so a
	// caller layering its own WithExtraFields on top still wins.
	if extra := passthroughRequestFields(body); len(extra) > 0 {
		opts = append(opts, llm.WithExtraFields(extra))
	}

	return model, stream, opts, nil
}

// ConvertOpenAIMessagesJSON converts a JSON array of OpenAI-style chat
// messages into genai's internal []llm.Message representation.
func ConvertOpenAIMessagesJSON(messagesJSON json.RawMessage) ([]llm.Message, error) {
	var msgs []openAIMessage
	if err := json.Unmarshal(messagesJSON, &msgs); err != nil {
		return nil, errors.Wrap(err, "could not unmarshal messages")
	}

	return convertMessages(msgs)
}

func convertMessages(msgs []openAIMessage) ([]llm.Message, error) {
	out := make([]llm.Message, 0, len(msgs))

	for _, m := range msgs {
		switch m.Role {
		case "tool":
			content := extractTextContent(m.Content)
			msg := llm.NewToolMessage(m.ToolCallID, llm.NewToolResult(content))
			out = append(out, msg)

		case "assistant":
			content := extractTextContent(m.Content)
			details := toLLMReasoningDetails(m.ReasoningDetails)
			hasReasoning := m.ReasoningContent != "" || len(details) > 0
			if len(m.ToolCalls) > 0 {
				calls := make([]llm.ToolCall, 0, len(m.ToolCalls))
				for _, tc := range m.ToolCalls {
					calls = append(calls, llm.NewToolCall(tc.ID, tc.Function.Name, tc.Function.Arguments))
				}
				if hasReasoning {
					out = append(out, llm.NewReasoningToolCallsMessageWithContent(content, m.ReasoningContent, details, calls...))
				} else {
					out = append(out, llm.NewToolCallsMessageWithContent(content, calls...))
				}
			} else if hasReasoning {
				out = append(out, llm.NewAssistantReasoningMessage(content, m.ReasoningContent, details))
			} else {
				out = append(out, llm.NewMessage(llm.RoleAssistant, content))
			}

		default:
			role := llm.Role(m.Role)
			text, attachments, cacheControl, err := extractContentParts(m.Content)
			if err != nil {
				return nil, errors.Wrapf(err, "could not convert content parts for role %s", m.Role)
			}
			if len(attachments) > 0 {
				out = append(out, llm.NewMultimodalMessage(role, text, attachments...))
			} else if cacheControl != nil {
				out = append(out, llm.NewMessageWithCacheControl(role, text, cacheControl))
			} else {
				out = append(out, llm.NewMessage(role, text))
			}
		}
	}

	return out, nil
}

// extractTextContent returns only the text from a message content field.
func extractTextContent(raw any) string {
	if raw == nil {
		return ""
	}
	switch v := raw.(type) {
	case string:
		return v
	case []any:
		var buf strings.Builder
		for _, part := range v {
			if partMap, ok := part.(map[string]any); ok {
				if t, ok := partMap["text"].(string); ok {
					buf.WriteString(t)
				}
			}
		}
		return buf.String()
	default:
		return fmt.Sprintf("%v", raw)
	}
}

// extractContentParts extracts text and attachments from a message content field.
// For string content it returns the string with no attachments.
//
// For array content it accepts every content-part dialect an OpenAI-compatible
// client may realistically emit, since clients differ widely in how they encode
// binary payloads:
//
//   - Chat Completions: "text", "image_url", "input_audio", "file"
//   - Responses API: "input_text", "output_text", "input_image", "input_file"
//   - Vercel AI SDK: "file" with "data"/"mediaType"
//   - Anthropic-shaped blocks ("image"/"document" with a "source" object) sent
//     to the OpenAI endpoint by clients that reuse a single message encoder
//
// A part whose type is unknown, or whose payload cannot be located, is skipped
// with a warning rather than silently dropped: a missing attachment is
// otherwise invisible to the caller and surfaces only as the model claiming it
// cannot see the file. A part that is recognised but malformed fails the whole
// request, so the client gets a real error instead of a degraded answer.
func extractContentParts(raw any) (text string, attachments []llm.Attachment, cacheControl *llm.CacheControl, err error) {
	if raw == nil {
		return "", nil, nil, nil
	}
	switch v := raw.(type) {
	case string:
		return v, nil, nil, nil
	case []any:
		var buf strings.Builder
		for i, part := range v {
			partMap, ok := part.(map[string]any)
			if !ok {
				// A bare string inside the array is a text part in some dialects.
				if s, ok := part.(string); ok {
					buf.WriteString(s)
					continue
				}
				slog.Warn("proxy: ignoring unsupported content part",
					slog.Int("index", i), slog.String("goType", fmt.Sprintf("%T", part)))
				continue
			}

			if cc := extractCacheControl(partMap["cache_control"]); cc != nil {
				cacheControl = cc
			}

			partType := partTypeOf(partMap)
			switch partType {
			case "text", "input_text", "output_text":
				if t, ok := partMap["text"].(string); ok {
					buf.WriteString(t)
				}

			case "image_url", "input_image", "image":
				att, attErr := convertImagePart(partMap)
				if attErr != nil {
					slog.Error("proxy: could not convert image content part",
						slog.Int("index", i), slog.String("type", partType), slog.Any("error", attErr))
					return "", nil, nil, errors.Wrapf(attErr, "could not convert %s content part", partType)
				}
				if att == nil {
					slog.Warn("proxy: image content part carries no usable payload, dropping it",
						slog.Int("index", i), slog.String("type", partType),
						slog.Any("keys", mapKeys(partMap)))
					continue
				}
				attachments = append(attachments, att)

			case "input_audio", "audio":
				att, attErr := convertAudioPart(partMap)
				if attErr != nil {
					slog.Error("proxy: could not convert audio content part",
						slog.Int("index", i), slog.String("type", partType), slog.Any("error", attErr))
					return "", nil, nil, errors.Wrapf(attErr, "could not convert %s content part", partType)
				}
				if att == nil {
					slog.Warn("proxy: audio content part carries no usable payload, dropping it",
						slog.Int("index", i), slog.String("type", partType),
						slog.Any("keys", mapKeys(partMap)))
					continue
				}
				attachments = append(attachments, att)

			case "file", "input_file", "document":
				att, inlined, attErr := convertFilePart(partMap)
				if attErr != nil {
					slog.Error("proxy: could not convert file content part",
						slog.Int("index", i), slog.String("type", partType), slog.Any("error", attErr))
					return "", nil, nil, errors.Wrapf(attErr, "could not convert %s content part", partType)
				}
				switch {
				case att != nil:
					attachments = append(attachments, att)
				case inlined != "":
					buf.WriteString(inlined)
				default:
					// A file referenced by provider-side id (e.g. "file_id") has no
					// payload we can forward — the upstream provider would have to
					// resolve it, which the proxy cannot do on the client's behalf.
					slog.Warn("proxy: file content part carries no inlinable payload, dropping it",
						slog.Int("index", i), slog.String("type", partType),
						slog.Any("keys", mapKeys(partMap)))
				}

			case "refusal", "thinking", "redacted_thinking", "tool_use", "tool_result":
				// Structural blocks with no textual payload to forward here; the
				// caller handles them (or they are meaningless upstream).
				slog.Debug("proxy: skipping structural content part",
					slog.Int("index", i), slog.String("type", partType))

			default:
				slog.Warn("proxy: unsupported content part type, dropping it",
					slog.Int("index", i), slog.String("type", partType),
					slog.Any("keys", mapKeys(partMap)))
			}
		}
		return buf.String(), attachments, cacheControl, nil
	default:
		return fmt.Sprintf("%v", raw), nil, nil, nil
	}
}

// partTypeOf returns the declared "type" of a content part, falling back to
// inferring it from the part's shape when the field is absent — some clients
// omit it for single-kind arrays.
func partTypeOf(part map[string]any) string {
	if t, ok := part["type"].(string); ok && t != "" {
		return t
	}
	switch {
	case part["image_url"] != nil:
		return "image_url"
	case part["input_audio"] != nil:
		return "input_audio"
	case part["source"] != nil:
		return "image"
	case part["text"] != nil:
		return "text"
	case part["file"] != nil, part["file_data"] != nil, part["data"] != nil:
		return "file"
	default:
		return "unknown"
	}
}

// mapKeys returns the keys of a content part, for diagnostics. Values are never
// logged: they may hold megabytes of base64 payload.
func mapKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// firstString returns the first non-empty string found under the given keys.
func firstString(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if s, ok := m[k].(string); ok && s != "" {
			return s
		}
	}
	return ""
}

// declaredMIMEType reads the MIME type of a part under any of the spellings in
// use across clients (mediaType, media_type, mimeType, mime_type, mime).
func declaredMIMEType(m map[string]any) string {
	return firstString(m, "mediaType", "media_type", "mimeType", "mime_type", "mime", "content_type", "contentType")
}

// convertImagePart builds an image attachment from any of the known image part
// encodings. It returns (nil, nil) when the part holds no usable payload.
func convertImagePart(part map[string]any) (llm.Attachment, error) {
	// Anthropic-shaped block: {"type":"image","source":{...}}
	if source, ok := part["source"].(map[string]any); ok {
		return convertAnthropicSource(llm.AttachmentTypeImage, source)
	}

	declared := declaredMIMEType(part)

	// Chat Completions: {"image_url":{"url":"...","detail":"..."}}
	if imgURL, ok := part["image_url"].(map[string]any); ok {
		if declared == "" {
			declared = declaredMIMEType(imgURL)
		}
		value := firstString(imgURL, "url", "data")
		if value == "" {
			return nil, nil
		}
		return newPayloadAttachment(llm.AttachmentTypeImage, value, declared)
	}

	// Responses API and shorthand variants: the payload sits directly on the part,
	// either as {"image_url":"data:..."} or {"data":"..."}/{"url":"..."}.
	value := firstString(part, "image_url", "url", "data", "image", "b64_json", "file_data")
	if value == "" {
		return nil, nil
	}
	return newPayloadAttachment(llm.AttachmentTypeImage, value, declared)
}

// convertAudioPart builds an audio attachment from the known audio encodings.
func convertAudioPart(part map[string]any) (llm.Attachment, error) {
	if source, ok := part["source"].(map[string]any); ok {
		return convertAnthropicSource(llm.AttachmentTypeAudio, source)
	}

	container := part
	if audio, ok := part["input_audio"].(map[string]any); ok {
		container = audio
	} else if audio, ok := part["audio"].(map[string]any); ok {
		container = audio
	}

	value := firstString(container, "data", "url", "audio_url")
	if value == "" {
		return nil, nil
	}

	mimeType := declaredMIMEType(container)
	if mimeType == "" {
		// OpenAI encodes the codec as a bare "format" field ("wav", "mp3"…).
		if format := firstString(container, "format"); format != "" {
			mimeType = "audio/" + format
		}
	}
	return newPayloadAttachment(llm.AttachmentTypeAudio, value, mimeType)
}

// convertFilePart builds an attachment from a file part. Text payloads are
// returned as inlined text instead, since no provider accepts them as binary
// attachments. Returns (nil, "", nil) when the part holds no forwardable
// payload (e.g. a provider-side file id).
func convertFilePart(part map[string]any) (att llm.Attachment, inlined string, err error) {
	if source, ok := part["source"].(map[string]any); ok {
		// Anthropic "document" block with a plain-text source.
		if source["type"] == "text" {
			t, _ := source["data"].(string)
			return nil, t, nil
		}
		a, err := convertAnthropicSource(llm.AttachmentTypeDocument, source)
		if err != nil {
			return nil, "", err
		}
		return a, "", nil
	}

	container := part
	if file, ok := part["file"].(map[string]any); ok {
		container = file
	}

	declared := declaredMIMEType(container)
	if declared == "" {
		declared = declaredMIMEType(part)
	}

	value := firstString(container, "file_data", "data", "url", "file_url")
	if value == "" {
		// Nothing but a filename/file_id: unusable without provider-side lookup.
		return nil, "", nil
	}

	// A filename is often the only hint about the payload's nature.
	if declared == "" {
		if filename := firstString(container, "filename", "name"); filename != "" {
			declared = mime.TypeByExtension(path.Ext(filename))
		}
	}

	a, err := newPayloadAttachment(attachmentTypeForMIME(declared), value, declared)
	if err != nil {
		return nil, "", err
	}
	return a, "", nil
}

// attachmentTypeForMIME maps a MIME type to the attachment type it belongs to,
// defaulting to document for anything non-media.
func attachmentTypeForMIME(mimeType string) llm.AttachmentType {
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		return llm.AttachmentTypeImage
	case strings.HasPrefix(mimeType, "audio/"):
		return llm.AttachmentTypeAudio
	case strings.HasPrefix(mimeType, "video/"):
		return llm.AttachmentTypeVideo
	default:
		return llm.AttachmentTypeDocument
	}
}

// newPayloadAttachment builds an attachment from a payload that may be a data
// URL, a remote URL or a bare base64 blob. declared is the MIME type announced
// by the client, if any; it is used when the payload carries none.
func newPayloadAttachment(attachmentType llm.AttachmentType, value, declared string) (llm.Attachment, error) {
	if withoutScheme, ok := strings.CutPrefix(value, "data:"); ok {
		meta, data, ok := strings.Cut(withoutScheme, ",")
		if !ok {
			return nil, errors.New("invalid data URL: missing comma")
		}

		mimeType := normalizeMIMEType(firstNonEmpty(mimeFromDataURLMeta(meta), declared))
		if mimeType == "" {
			return nil, errors.New("invalid data URL: missing media type")
		}

		// A data URL without the ";base64" flag holds percent-encoded text.
		if !dataURLIsBase64(meta) {
			decoded, err := neturl.PathUnescape(data)
			if err != nil {
				return nil, errors.Wrap(err, "invalid data URL: malformed percent encoding")
			}
			data = base64.StdEncoding.EncodeToString([]byte(decoded))
		}

		return llm.NewBase64Attachment(attachmentType, mimeType, data)
	}

	if isRemoteURL(value) {
		return llm.NewURLAttachment(attachmentType, remoteMIMEType(declared, value, attachmentType), value)
	}

	// Bare payload: treat it as raw base64, which is how the Vercel AI SDK and
	// several clients encode "file" parts.
	mimeType := normalizeMIMEType(declared)
	if mimeType == "" {
		return nil, errors.Errorf("attachment payload has no media type and none could be inferred")
	}
	return llm.NewBase64Attachment(attachmentType, mimeType, value)
}

// mimeFromDataURLMeta extracts the media type from a data URL's metadata
// segment ("image/png;base64" → "image/png").
func mimeFromDataURLMeta(meta string) string {
	mimeType, _, _ := strings.Cut(meta, ";")
	return strings.TrimSpace(mimeType)
}

func dataURLIsBase64(meta string) bool {
	for _, param := range strings.Split(meta, ";")[1:] {
		if strings.EqualFold(strings.TrimSpace(param), "base64") {
			return true
		}
	}
	return false
}

// isRemoteURL reports whether value looks like an absolute URL the provider can
// fetch, as opposed to a base64 blob.
func isRemoteURL(value string) bool {
	parsed, err := neturl.Parse(value)
	return err == nil && parsed.Scheme != "" && parsed.Host != ""
}

// remoteMIMEType determines the media type to attach to a remote URL. The
// client rarely declares one, so it is inferred from the URL's extension; when
// that fails a conventional default is assumed, because llm.Attachment requires
// a syntactically valid type and providers fetching the URL determine the real
// one themselves.
func remoteMIMEType(declared, url string, attachmentType llm.AttachmentType) string {
	if mimeType := normalizeMIMEType(declared); mimeType != "" && mimeType != "image/*" {
		return mimeType
	}

	trimmed := url
	if idx := strings.IndexAny(trimmed, "?#"); idx >= 0 {
		trimmed = trimmed[:idx]
	}
	if ext := path.Ext(trimmed); ext != "" {
		if mimeType, _, err := mime.ParseMediaType(mime.TypeByExtension(ext)); err == nil && mimeType != "" {
			return mimeType
		}
	}

	fallback := defaultMIMETypes[attachmentType]
	slog.Warn("proxy: could not determine media type of remote attachment, assuming a default",
		slog.String("url", url), slog.String("assumed", fallback))
	return fallback
}

var defaultMIMETypes = map[llm.AttachmentType]string{
	llm.AttachmentTypeImage:    "image/jpeg",
	llm.AttachmentTypeAudio:    "audio/mpeg",
	llm.AttachmentTypeVideo:    "video/mp4",
	llm.AttachmentTypeDocument: "application/octet-stream",
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// extractCacheControl parses an optional cache_control annotation from a
// content part, expecting a map with a "type" string and optional "ttl" string.
func extractCacheControl(raw any) *llm.CacheControl {
	ccMap, ok := raw.(map[string]any)
	if !ok {
		return nil
	}
	ccType, ok := ccMap["type"].(string)
	if !ok || ccType == "" {
		return nil
	}
	cc := &llm.CacheControl{Type: ccType}
	if ttl, ok := ccMap["ttl"].(string); ok && ttl != "" {
		cc.TTL = &ttl
	}
	return cc
}

// convertImageURL creates an llm.Attachment from an OpenAI image_url value.
// It handles both data URLs (data:mime;base64,<data>) and regular https:// URLs.
func convertImageURL(url string) (llm.Attachment, error) {
	return newPayloadAttachment(llm.AttachmentTypeImage, url, "")
}

// normalizeMIMEType maps common non-standard MIME type aliases to their canonical forms.
var mimeTypeAliases = map[string]string{
	"image/jpg":  "image/jpeg",
	"image/jpe":  "image/jpeg",
	"image/tiff": "image/tiff",
}

// normalizeMIMEType canonicalises a client-provided media type, dropping any
// parameters ("image/png; charset=binary") and wildcards, which are not valid
// attachment media types.
func normalizeMIMEType(mimeType string) string {
	mimeType = strings.ToLower(strings.TrimSpace(mimeType))
	if mimeType == "" {
		return ""
	}
	if parsed, _, err := mime.ParseMediaType(mimeType); err == nil && parsed != "" {
		mimeType = parsed
	} else if base, _, found := strings.Cut(mimeType, ";"); found {
		mimeType = strings.TrimSpace(base)
	}
	if canonical, ok := mimeTypeAliases[mimeType]; ok {
		return canonical
	}
	if strings.HasSuffix(mimeType, "/*") {
		return ""
	}
	return mimeType
}

// FormatChatCompletionResponse converts a llm.ChatCompletionResponse to OpenAI JSON.
func FormatChatCompletionResponse(res llm.ChatCompletionResponse, model string) any {
	msg := res.Message()

	oMsg := openAIMessage{
		Role:    string(msg.Role()),
		Content: msg.Content(),
	}

	if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok {
		oMsg.ReasoningContent = rr.Reasoning()
		oMsg.ReasoningDetails = fromLLMReasoningDetails(rr.ReasoningDetails())
	}

	finishReason := "stop"

	if len(res.ToolCalls()) > 0 {
		finishReason = "tool_calls"
		tcs := make([]openAIToolCall, 0, len(res.ToolCalls()))
		for _, tc := range res.ToolCalls() {
			args := ""
			switch p := tc.Parameters().(type) {
			case string:
				args = p
			default:
				raw, _ := json.Marshal(p)
				args = string(raw)
			}
			tcs = append(tcs, openAIToolCall{
				ID:   tc.ID(),
				Type: "function",
				Function: openAIFunctionCall{
					Name:      tc.Name(),
					Arguments: args,
				},
			})
		}
		oMsg.ToolCalls = tcs
		// Keep any text emitted alongside the tool calls: clients replay the
		// assistant turn verbatim, and blanking it here strips the rationale
		// from the history the model sees on the next turn.
		if msg.Content() == "" {
			oMsg.Content = nil
		}
	}

	usage := res.Usage()

	return openAIChatResponse{
		ID:      "chatcmpl-" + uuid.New().String(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openAIChoice{
			{
				Index:        0,
				Message:      oMsg,
				FinishReason: finishReason,
			},
		},
		Usage: openAIUsage{
			PromptTokens:     usage.PromptTokens(),
			CompletionTokens: usage.CompletionTokens(),
			TotalTokens:      usage.TotalTokens(),
		},
	}
}

// FormatStreamChunk converts a llm.StreamChunk to an OpenAI SSE data payload.
//
// sawToolCalls must be true if any prior chunk in the same stream carried tool
// calls. It selects the finish_reason emitted on the terminal chunk:
//   - tool_calls stream → finish_reason:"tool_calls"
//   - text stream       → finish_reason:"stop"
//
// finish_reason is only ever emitted on the terminal chunk. Tool call arguments
// are streamed across several deltas, and OpenAI-compatible clients treat the
// first non-null finish_reason as the end of the turn: setting it on an
// intermediate tool-call chunk makes them close the turn on truncated
// arguments, then retry the call — an agent loop that repeats the same tools.
func FormatStreamChunk(chunk llm.StreamChunk, id, model string, sawToolCalls bool) any {
	if chunk.IsComplete() {
		c := openAIStreamChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   model,
		}
		finish := "stop"
		if sawToolCalls {
			finish = "tool_calls"
		}
		c.Choices = []openAIStreamChoice{
			{Index: 0, Delta: openAIStreamDelta{}, FinishReason: &finish},
		}
		if usage := chunk.Usage(); usage != nil {
			c.Usage = &openAIUsage{
				PromptTokens:     usage.PromptTokens(),
				CompletionTokens: usage.CompletionTokens(),
				TotalTokens:      usage.TotalTokens(),
			}
		}
		return c
	}

	delta := openAIStreamDelta{}
	if d := chunk.Delta(); d != nil {
		delta.Content = d.Content()
		if string(d.Role()) != "" {
			delta.Role = string(d.Role())
		}

		if rd, ok := d.(llm.ReasoningStreamDelta); ok {
			delta.ReasoningContent = rd.Reasoning()
			delta.ReasoningDetails = fromLLMReasoningDetails(rd.ReasoningDetails())
		}

		if len(d.ToolCalls()) > 0 {
			tcs := make([]openAIStreamToolCall, 0, len(d.ToolCalls()))
			for _, tc := range d.ToolCalls() {
				// Use the provider's index, not the position in this chunk:
				// clients key their accumulator on it, so renumbering per
				// chunk merges parallel tool calls into one and concatenates
				// their arguments into invalid JSON.
				tcs = append(tcs, openAIStreamToolCall{
					Index: tc.Index(),
					ID:    tc.ID(),
					Type:  "function",
					Function: openAIFunctionCall{
						Name:      tc.Name(),
						Arguments: tc.ParametersDelta(),
					},
				})
			}
			delta.ToolCalls = tcs
		}
	}

	c := openAIStreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openAIStreamChoice{
			{
				Index: 0,
				Delta: delta,
				// Intentionally nil: finish_reason belongs to the terminal
				// chunk only (see the doc comment above).
				FinishReason: nil,
			},
		},
	}

	if usage := chunk.Usage(); usage != nil {
		c.Usage = &openAIUsage{
			PromptTokens:     usage.PromptTokens(),
			CompletionTokens: usage.CompletionTokens(),
			TotalTokens:      usage.TotalTokens(),
		}
	}

	return c
}

// ParseEmbeddingRequest converts an OpenAI embedding request body to llm options and inputs.
func ParseEmbeddingRequest(body json.RawMessage) (model string, inputs []string, opts []llm.EmbeddingsOptionFunc, err error) {
	var req openAIEmbeddingRequest
	if err = json.Unmarshal(body, &req); err != nil {
		return "", nil, nil, errors.Wrap(err, "could not parse embedding request")
	}

	model = req.Model

	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []any:
		for _, item := range v {
			if s, ok := item.(string); ok {
				inputs = append(inputs, s)
			}
		}
	default:
		return "", nil, nil, errors.New("invalid input type in embedding request")
	}

	if req.Dimensions != nil {
		opts = append(opts, llm.WithDimensions(*req.Dimensions))
	}

	return model, inputs, opts, nil
}

// FormatEmbeddingResponse converts a llm.EmbeddingsResponse to OpenAI JSON.
func FormatEmbeddingResponse(res llm.EmbeddingsResponse, model string) any {
	data := make([]openAIEmbeddingObj, 0, len(res.Embeddings()))
	for i, emb := range res.Embeddings() {
		data = append(data, openAIEmbeddingObj{
			Index:     i,
			Object:    "embedding",
			Embedding: emb,
		})
	}

	usage := res.Usage()

	return openAIEmbeddingResponse{
		Object: "list",
		Model:  model,
		Data:   data,
		Usage: openAIEmbeddingUsage{
			PromptTokens: usage.PromptTokens(),
			TotalTokens:  usage.TotalTokens(),
		},
	}
}

// FormatModelsResponse converts a list of ModelInfo to OpenAI JSON.
func FormatModelsResponse(models []ModelInfo) any {
	data := make([]openAIModelObj, 0, len(models))
	for _, m := range models {
		data = append(data, openAIModelObj{
			ID:      m.ID,
			Object:  "model",
			Created: m.Created,
			OwnedBy: m.OwnedBy,
		})
	}
	return openAIModelsResponse{
		Object: "list",
		Data:   data,
	}
}

package proxy

import (
	"encoding/json"
	"fmt"
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
}

type openAITool struct {
	Type     string          `json:"type"` // "function"
	Function openAIFunction  `json:"function"`
}

type openAIFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type openAIToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"` // "function"
	Function openAIFunctionCall  `json:"function"`
}

type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ---- OpenAI response types ----------------------------------------------

type openAIChatResponse struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []openAIChoice    `json:"choices"`
	Usage   openAIUsage       `json:"usage"`
}

type openAIChoice struct {
	Index        int            `json:"index"`
	Message      openAIMessage  `json:"message"`
	FinishReason string         `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// openAIStreamChunk is a single SSE data payload.
type openAIStreamChunk struct {
	ID      string                `json:"id"`
	Object  string                `json:"object"`
	Created int64                 `json:"created"`
	Model   string                `json:"model"`
	Choices []openAIStreamChoice  `json:"choices"`
	Usage   *openAIUsage          `json:"usage,omitempty"`
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
		}
	}

	return model, stream, opts, nil
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
			if len(m.ToolCalls) > 0 {
				calls := make([]llm.ToolCall, 0, len(m.ToolCalls))
				for _, tc := range m.ToolCalls {
					calls = append(calls, llm.NewToolCall(tc.ID, tc.Function.Name, tc.Function.Arguments))
				}
				if m.ReasoningContent != "" {
					out = append(out, llm.NewReasoningToolCallsMessage(m.ReasoningContent, nil, calls...))
				} else {
					out = append(out, llm.NewToolCallsMessage(calls...))
				}
			} else if m.ReasoningContent != "" {
				out = append(out, llm.NewAssistantReasoningMessage(content, m.ReasoningContent, nil))
			} else {
				out = append(out, llm.NewMessage(llm.RoleAssistant, content))
			}

		default:
			role := llm.Role(m.Role)
			text, attachments, err := extractContentParts(m.Content)
			if err != nil {
				return nil, errors.Wrapf(err, "could not convert content parts for role %s", m.Role)
			}
			if len(attachments) > 0 {
				out = append(out, llm.NewMultimodalMessage(role, text, attachments...))
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
// For array content it parses text and image_url parts.
func extractContentParts(raw any) (text string, attachments []llm.Attachment, err error) {
	if raw == nil {
		return "", nil, nil
	}
	switch v := raw.(type) {
	case string:
		return v, nil, nil
	case []any:
		var buf strings.Builder
		for _, part := range v {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}
			switch partMap["type"] {
			case "text":
				if t, ok := partMap["text"].(string); ok {
					buf.WriteString(t)
				}
			case "image_url":
				imgURL, ok := partMap["image_url"].(map[string]any)
				if !ok {
					continue
				}
				url, _ := imgURL["url"].(string)
				if url == "" {
					continue
				}
				att, attErr := convertImageURL(url)
				if attErr != nil {
					return "", nil, errors.Wrapf(attErr, "could not convert image_url")
				}
				attachments = append(attachments, att)
			case "input_audio":
				audio, ok := partMap["input_audio"].(map[string]any)
				if !ok {
					continue
				}
				data, _ := audio["data"].(string)
				format, _ := audio["format"].(string)
				if data == "" {
					continue
				}
				mimeType := "audio/" + format
				att, attErr := llm.NewBase64Attachment(llm.AttachmentTypeAudio, mimeType, data)
				if attErr != nil {
					return "", nil, errors.Wrapf(attErr, "could not convert input_audio")
				}
				attachments = append(attachments, att)
			}
		}
		return buf.String(), attachments, nil
	default:
		return fmt.Sprintf("%v", raw), nil, nil
	}
}

// convertImageURL creates an llm.Attachment from an OpenAI image_url value.
// It handles both data URLs (data:mime;base64,<data>) and regular https:// URLs.
func convertImageURL(url string) (llm.Attachment, error) {
	if withoutScheme, ok := strings.CutPrefix(url, "data:"); ok {
		// Format: data:<mimeType>;base64,<data>
		meta, data, ok := strings.Cut(withoutScheme, ",")
		if !ok {
			return nil, errors.New("invalid data URL: missing comma")
		}

		mimeType, _, _ := strings.Cut(meta, ";")
		mimeType = normalizeMIMEType(mimeType)

		return llm.NewBase64Attachment(llm.AttachmentTypeImage, mimeType, data)
	}

	// Regular URL — MIME type unknown, use generic image/* placeholder.
	return llm.NewImageAttachment("image/*", url, true)
}

// normalizeMIMEType maps common non-standard MIME type aliases to their canonical forms.
var mimeTypeAliases = map[string]string{
	"image/jpg":  "image/jpeg",
	"image/jpe":  "image/jpeg",
	"image/tiff": "image/tiff",
}

func normalizeMIMEType(mimeType string) string {
	if canonical, ok := mimeTypeAliases[strings.ToLower(mimeType)]; ok {
		return canonical
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
		oMsg.Content = nil
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
// calls. It is used to decide the finish_reason on the terminal chunk:
//   - tool_calls stream → empty choices, usage only (finish_reason already sent)
//   - text stream       → finish_reason:"stop" so clients know the turn is done
func FormatStreamChunk(chunk llm.StreamChunk, id, model string, sawToolCalls bool) any {
	if chunk.IsComplete() {
		c := openAIStreamChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   model,
		}
		if sawToolCalls {
			// finish_reason:"tool_calls" was already sent on the tool-call chunk;
			// emit usage-only with empty choices to avoid overwriting it.
			c.Choices = []openAIStreamChoice{}
		} else {
			// Text response: emit finish_reason:"stop" so the client knows the
			// turn is complete.
			stop := "stop"
			c.Choices = []openAIStreamChoice{
				{Index: 0, Delta: openAIStreamDelta{}, FinishReason: &stop},
			}
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

	finishReason := (*string)(nil)

	delta := openAIStreamDelta{}
	if d := chunk.Delta(); d != nil {
		delta.Content = d.Content()
		if string(d.Role()) != "" {
			delta.Role = string(d.Role())
		}

		if rd, ok := d.(llm.ReasoningStreamDelta); ok {
			delta.ReasoningContent = rd.Reasoning()
		}

		if len(d.ToolCalls()) > 0 {
			tcs := make([]openAIStreamToolCall, 0, len(d.ToolCalls()))
			for i, tc := range d.ToolCalls() {
				tcs = append(tcs, openAIStreamToolCall{
					Index: i,
					ID:    tc.ID(),
					Type:  "function",
					Function: openAIFunctionCall{
						Name:      tc.Name(),
						Arguments: tc.ParametersDelta(),
					},
				})
			}
			delta.ToolCalls = tcs
			fr := "tool_calls"
			finishReason = &fr
		}
	}

	c := openAIStreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openAIStreamChoice{
			{
				Index:        0,
				Delta:        delta,
				FinishReason: finishReason,
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

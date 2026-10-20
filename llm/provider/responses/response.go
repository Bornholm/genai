package responses

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/bornholm/genai/llm"
)

// response is a Responses API response object, reduced to what genai reads.
type response struct {
	ID                string             `json:"id"`
	Status            string             `json:"status"`
	Error             *responseError     `json:"error"`
	IncompleteDetails *incompleteDetails `json:"incomplete_details"`
	Output            []outputItem       `json:"output"`
	Usage             *responseUsage     `json:"usage"`
}

type responseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type incompleteDetails struct {
	Reason string `json:"reason"`
}

// outputItem is one item of a response's output.
type outputItem struct {
	Type string `json:"type"`
	ID   string `json:"id"`

	// message, and reasoning for models returning their raw reasoning text
	Content []outputContent `json:"content"`

	// reasoning
	Summary          []outputContent `json:"summary"`
	EncryptedContent string          `json:"encrypted_content"`

	// function_call
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type outputContent struct {
	Type    string `json:"type"`
	Text    string `json:"text"`
	Refusal string `json:"refusal"`
}

type responseUsage struct {
	InputTokens        int64 `json:"input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	InputTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
}

// newUsage converts the usage of a response. input_tokens already counts the
// cached prefix, as genai's PromptTokens does.
func newUsage(u *responseUsage) llm.ChatCompletionUsage {
	if u == nil {
		return llm.NewChatCompletionUsage(0, 0, 0)
	}
	total := u.TotalTokens
	if total == 0 {
		total = u.InputTokens + u.OutputTokens
	}
	return llm.NewChatCompletionUsageWithCache(u.InputTokens, u.OutputTokens, total, u.InputTokensDetails.CachedTokens)
}

// usagePublished reports whether a usage carries any counter.
func usagePublished(u *responseUsage) bool {
	return u != nil && (u.InputTokens > 0 || u.OutputTokens > 0 || u.TotalTokens > 0 || u.InputTokensDetails.CachedTokens > 0)
}

// messageText returns the text of a message item. A refusal is the model's
// answer as much as a text is, so it is returned as content.
func messageText(item outputItem) string {
	var text strings.Builder
	for _, c := range item.Content {
		switch c.Type {
		case "output_text":
			text.WriteString(c.Text)
		case "refusal":
			text.WriteString(c.Refusal)
		}
	}
	return text.String()
}

// reasoningText returns the readable reasoning of a reasoning item: the raw
// reasoning text when the model returns it, its summaries otherwise, one
// paragraph each.
func reasoningText(item outputItem) string {
	var parts []string
	for _, c := range item.Content {
		if c.Type == "reasoning_text" && c.Text != "" {
			parts = append(parts, c.Text)
		}
	}
	if len(parts) == 0 {
		for _, s := range item.Summary {
			if s.Text != "" {
				parts = append(parts, s.Text)
			}
		}
	}
	return strings.Join(parts, "\n\n")
}

// textPart is one part of readable reasoning and its text.
type textPart struct {
	part reasoningPart
	text string
}

// reasoningParts returns the parts reasoningText joins, each with the part
// its deltas stream under.
func reasoningParts(outputIndex int, item outputItem) []textPart {
	var parts []textPart
	for i, c := range item.Content {
		if c.Type == "reasoning_text" && c.Text != "" {
			parts = append(parts, textPart{reasoningPart{outputIndex: outputIndex, index: i, raw: true}, c.Text})
		}
	}
	if len(parts) > 0 {
		return parts
	}
	for i, s := range item.Summary {
		if s.Text != "" {
			parts = append(parts, textPart{reasoningPart{outputIndex: outputIndex, index: i}, s.Text})
		}
	}
	return parts
}

// reasoningDetails returns the details that let a later turn replay a
// reasoning item: one summary detail per summary part and the encrypted
// content, all carrying the item ID. index is the index of the first one.
func reasoningDetails(item outputItem, index int) []llm.ReasoningDetail {
	var details []llm.ReasoningDetail
	for _, s := range item.Summary {
		if s.Text == "" {
			continue
		}
		details = append(details, llm.ReasoningDetail{
			ID:      item.ID,
			Type:    llm.ReasoningDetailTypeSummary,
			Summary: s.Text,
			Format:  reasoningFormat,
			Index:   index + len(details),
		})
	}
	if item.EncryptedContent != "" {
		details = append(details, llm.ReasoningDetail{
			ID:     item.ID,
			Type:   llm.ReasoningDetailTypeEncrypted,
			Data:   item.EncryptedContent,
			Format: reasoningFormat,
			Index:  index + len(details),
		})
	}
	return details
}

// fromResponse converts a complete response. It returns llm.ErrNoMessage
// when the output holds neither a message nor a function call.
func fromResponse(res *response, excludeReasoning bool) (llm.ChatCompletionResponse, error) {
	if err := responseFailure(res); err != nil {
		return nil, err
	}

	var (
		content   strings.Builder
		reasoning []string
		details   []llm.ReasoningDetail
		toolCalls []llm.ToolCall
		answered  bool
	)

	for _, item := range res.Output {
		switch item.Type {
		case "message":
			if text := messageText(item); text != "" {
				answered = true
				content.WriteString(text)
			}
		case "reasoning":
			if text := reasoningText(item); text != "" {
				reasoning = append(reasoning, text)
			}
			details = append(details, reasoningDetails(item, len(details))...)
		case "function_call":
			answered = true
			toolCalls = append(toolCalls, llm.NewToolCall(item.CallID, item.Name, item.Arguments))
		}
	}

	if !answered {
		return nil, llm.ErrNoMessage
	}

	usage := newUsage(res.Usage)

	if len(reasoning) == 0 && len(details) == 0 {
		return llm.NewChatCompletionResponse(llm.NewMessage(llm.RoleAssistant, content.String()), usage, toolCalls...), nil
	}

	// The details stay whatever the caller asked: they are what the next
	// turn replays. Excluding only hides the readable text.
	text := strings.Join(reasoning, "\n\n")
	if excludeReasoning {
		text = ""
	}

	return llm.NewChatCompletionResponseWithReasoning(
		llm.NewAssistantReasoningMessage(content.String(), text, details),
		usage,
		text,
		details,
		toolCalls...,
	), nil
}

// responseFailure returns the error of a failed response, nil otherwise. A
// gateway may answer a 200 carrying only an error object, without a status:
// the error is the failure then too. An incomplete response is not a
// failure: like a chat completion cut by its length limit, it returns what
// was produced.
func responseFailure(res *response) error {
	failed := res.Status == "failed" || (res.Error != nil && (res.Error.Code != "" || res.Error.Message != ""))
	if !failed {
		return nil
	}
	code, message := "", "the response failed"
	if res.Error != nil {
		code, message = res.Error.Code, res.Error.Message
	}
	return failureError(code, message)
}

// failureError builds the error for a failure the API reports in a response
// or an error event, with the status the code stands for, so that the retry
// machinery tells a transient failure from a rejected request.
func failureError(code, message string) error {
	body, _ := json.Marshal(map[string]any{"error": map[string]string{"code": code, "message": message}})
	return llm.RateLimitError(statusForCode(code), string(body))
}

func statusForCode(code string) int {
	switch code {
	case "rate_limit_exceeded":
		return http.StatusTooManyRequests
	case "insufficient_quota":
		return http.StatusTooManyRequests
	case "invalid_prompt", "invalid_request_error", "context_length_exceeded",
		"invalid_image", "invalid_image_format", "invalid_base64_image", "invalid_image_url",
		"image_too_large", "image_too_small", "image_parse_error", "image_content_policy_violation",
		"invalid_image_mode", "image_file_too_large", "unsupported_image_media_type",
		"empty_image_file", "failed_to_download_image", "image_file_not_found":
		return http.StatusBadRequest
	case "vector_store_timeout":
		return http.StatusGatewayTimeout
	default:
		// server_error and anything new: transient until proven otherwise.
		return http.StatusBadGateway
	}
}

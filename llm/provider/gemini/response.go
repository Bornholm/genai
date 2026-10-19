package gemini

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/google/uuid"
)

// response is a generateContent response, or one chunk of a stream, reduced
// to what genai reads.
type response struct {
	Candidates     []candidate     `json:"candidates"`
	PromptFeedback *promptFeedback `json:"promptFeedback"`
	UsageMetadata  *usageMetadata  `json:"usageMetadata"`
	// Error is set on an error a stream reports in its body.
	Error *apiError `json:"error"`
}

type candidate struct {
	Content      *content `json:"content"`
	FinishReason string   `json:"finishReason"`
}

type promptFeedback struct {
	BlockReason string `json:"blockReason"`
}

type usageMetadata struct {
	PromptTokenCount        int64 `json:"promptTokenCount"`
	CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int64 `json:"thoughtsTokenCount"`
	CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
	TotalTokenCount         int64 `json:"totalTokenCount"`
}

type apiError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// newUsage converts the usage metadata. The thinking tokens are billed as
// output, so they count as completion tokens; promptTokenCount already
// includes the cached prefix, as genai's PromptTokens does.
func newUsage(u *usageMetadata) llm.ChatCompletionUsage {
	if u == nil {
		return llm.NewChatCompletionUsage(0, 0, 0)
	}
	completion := u.CandidatesTokenCount + u.ThoughtsTokenCount
	total := u.TotalTokenCount
	if total == 0 {
		total = u.PromptTokenCount + completion
	}
	return llm.NewChatCompletionUsageWithCache(u.PromptTokenCount, completion, total, u.CachedContentTokenCount)
}

func usagePublished(u *usageMetadata) bool {
	return u != nil && (u.PromptTokenCount > 0 || u.CandidatesTokenCount > 0 || u.ThoughtsTokenCount > 0 || u.TotalTokenCount > 0)
}

// finished reports whether a finish reason ends the answer normally.
func finished(reason string) bool {
	return reason == "STOP" || reason == "MAX_TOKENS"
}

// blockedError reports a prompt or an answer the model refused to produce.
// A safety block repeats on a retry, so it is not a 5xx; a malformed
// function call or an unexplained stop may not, so it is a 502.
func blockedError(what, reason string) error {
	status := http.StatusBadRequest
	switch reason {
	case "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "OTHER", "FINISH_REASON_UNSPECIFIED", "":
		status = http.StatusBadGateway
	}
	body, _ := json.Marshal(map[string]any{"error": map[string]string{"message": what + " blocked: " + reason, "status": reason}})
	return llm.RateLimitError(status, string(body))
}

// streamError converts an error a stream reports in its body.
func streamError(e *apiError) error {
	status := e.Code
	if status < http.StatusBadRequest {
		status = http.StatusBadGateway
	}
	body, _ := json.Marshal(map[string]any{"error": e})
	return llm.RateLimitError(status, string(body))
}

// newCallID names a function call the API returned without an ID: genai
// matches tool results to their call by ID.
func newCallID() string {
	return "call_" + strings.ReplaceAll(uuid.NewString(), "-", "")
}

// signatureDetail keeps a thought signature for the next turn, under the ID
// of the function call it came with, or an empty ID for a text part.
func signatureDetail(id, signature string, index int) llm.ReasoningDetail {
	return llm.ReasoningDetail{
		ID:     id,
		Type:   llm.ReasoningDetailTypeEncrypted,
		Data:   signature,
		Format: signatureFormat,
		Index:  index,
	}
}

// fromResponse converts a complete response.
func fromResponse(res *response, excludeReasoning bool) (llm.ChatCompletionResponse, error) {
	if res.PromptFeedback != nil && res.PromptFeedback.BlockReason != "" {
		return nil, blockedError("prompt", res.PromptFeedback.BlockReason)
	}
	if len(res.Candidates) == 0 || res.Candidates[0].Content == nil {
		if len(res.Candidates) > 0 && !finished(res.Candidates[0].FinishReason) {
			return nil, blockedError("answer", res.Candidates[0].FinishReason)
		}
		return nil, llm.ErrNoMessage
	}

	var (
		text      strings.Builder
		reasoning strings.Builder
		details   []llm.ReasoningDetail
		toolCalls []llm.ToolCall
		answered  bool
	)

	for _, p := range res.Candidates[0].Content.Parts {
		switch {
		case p.FunctionCall != nil:
			answered = true
			id := p.FunctionCall.ID
			if id == "" {
				id = newCallID()
			}
			args, _ := json.Marshal(p.FunctionCall.Args)
			toolCalls = append(toolCalls, llm.NewToolCall(id, p.FunctionCall.Name, string(args)))
			if p.ThoughtSignature != "" {
				details = append(details, signatureDetail(id, p.ThoughtSignature, len(details)))
			}
			continue
		case p.Thought:
			reasoning.WriteString(p.Text)
		default:
			if p.Text != "" {
				answered = true
			}
			text.WriteString(p.Text)
		}
		if p.ThoughtSignature != "" {
			details = append(details, signatureDetail("", p.ThoughtSignature, len(details)))
		}
	}

	if !answered {
		if reason := res.Candidates[0].FinishReason; !finished(reason) {
			return nil, blockedError("answer", reason)
		}
		return nil, llm.ErrNoMessage
	}

	usage := newUsage(res.UsageMetadata)

	if reasoning.Len() == 0 && len(details) == 0 {
		return llm.NewChatCompletionResponse(llm.NewMessage(llm.RoleAssistant, text.String()), usage, toolCalls...), nil
	}

	thoughts := reasoning.String()
	if excludeReasoning {
		thoughts = ""
	}

	return llm.NewChatCompletionResponseWithReasoning(
		llm.NewAssistantReasoningMessage(text.String(), thoughts, details),
		usage,
		thoughts,
		details,
		toolCalls...,
	), nil
}

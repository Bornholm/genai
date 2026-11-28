package openrouter

import (
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/revrost/go-openrouter"
)

// parseFinishReason translates the finish reason of a choice. OpenRouter
// normalizes finish_reason to the OpenAI vocabulary and keeps the upstream
// provider's own value in native_finish_reason, whose case varies by provider:
// the native value is the fallback when the normalized one is missing or
// unknown, so that a cut by the token limit or a filter still says so.
func parseFinishReason(finishReason openrouter.FinishReason, nativeFinishReason string) llm.FinishReason {
	if reason := llm.ParseOpenAIFinishReason(string(finishReason)); reason != "" {
		return reason
	}
	native := strings.ToLower(nativeFinishReason)
	if reason := llm.ParseOpenAIFinishReason(native); reason != "" {
		return reason
	}
	switch native {
	// The filtering reasons of Google's models, the upstream whose own
	// vocabulary differs most from OpenAI's.
	case "safety", "recitation", "blocklist", "prohibited_content", "spii", "image_safety", "language":
		return llm.FinishReasonContentFilter
	default:
		return ""
	}
}

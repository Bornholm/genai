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
// unknown, read the same way, so that a cut by the token limit still says so.
func parseFinishReason(finishReason openrouter.FinishReason, nativeFinishReason string) llm.FinishReason {
	if reason := llm.ParseOpenAIFinishReason(string(finishReason)); reason != "" {
		return reason
	}
	return llm.ParseOpenAIFinishReason(strings.ToLower(nativeFinishReason))
}

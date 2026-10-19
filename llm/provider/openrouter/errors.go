package openrouter

import (
	"net/http"
	"strings"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"github.com/revrost/go-openrouter"
)

// providerError turns an error of the OpenRouter SDK into an *llm.HTTPError
// when it carries an HTTP status, so that llm.IsRetryable sees a 429 or a 5xx.
// The SDK returns an *openrouter.APIError when the error body parses, which is
// the usual case, and an *openrouter.RequestError otherwise: both are mapped.
func providerError(err error) error {
	if status, ok := openrouter.HTTPStatusCode(err); ok {
		return errors.WithStack(llm.RateLimitError(status, err.Error()))
	}
	return errors.WithStack(err)
}

// drainTimeout bounds the read releasing the SDK reader of a stream stopped
// early, see ChatCompletionStream.
const drainTimeout = 30 * time.Second

// isFailure reports whether a choice ended on a provider failure, which
// OpenRouter gives as the finish reason "error". The native finish reason
// repeats the upstream provider's own value, whose case varies by provider.
func isFailure(finishReason openrouter.FinishReason, nativeFinishReason string) bool {
	return strings.EqualFold(string(finishReason), "error") || strings.EqualFold(nativeFinishReason, "error")
}

// errProviderFailed and errStreamTruncated report an answer that did not
// finish: the provider failed while producing it, the SDK having dropped the
// error event that carried the cause, or the stream was cut. They are 502s,
// which the gateway is acting as, so that the call is retried when nothing was
// forwarded yet. Each call builds its own: an *llm.HTTPError is mutable.
func errProviderFailed() error {
	return llm.NewHTTPError(http.StatusBadGateway, "openrouter: the provider failed while producing the answer")
}

func errStreamTruncated() error {
	return llm.NewHTTPError(http.StatusBadGateway, "openrouter: the stream ended before the model finished its answer")
}

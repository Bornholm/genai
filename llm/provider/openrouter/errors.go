package openrouter

import (
	"net/http"

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

// finishReasonError is the finish reason OpenRouter gives a stream the
// provider failed after it started: the error event carrying the cause is
// dropped by the SDK, the finish reason is all that is left of it.
const finishReasonError openrouter.FinishReason = "error"

// errStreamFailed and errStreamTruncated end a stream that did not finish.
// They are reported as a 502, which the gateway is acting as, so that the
// stream is retried when nothing was forwarded yet.
var (
	errStreamFailed    = llm.NewHTTPError(http.StatusBadGateway, "openrouter: the provider failed while streaming the answer")
	errStreamTruncated = llm.NewHTTPError(http.StatusBadGateway, "openrouter: the stream ended before the model finished its answer")
)

package proxy

import (
	"errors"
	"net/http"

	"github.com/bornholm/genai/llm"
)

// apiErrorFromErr maps a backend LLM error to an *APIError, preserving the
// upstream HTTP status code when the provider returned an *llm.HTTPError.
func apiErrorFromErr(err error) *APIError {
	var httpErr *llm.HTTPError
	if errors.As(err, &httpErr) {
		return &APIError{
			StatusCode: httpErr.StatusCode,
			Type:       httpErrorType(httpErr.StatusCode),
			Message:    httpErr.Error(),
		}
	}

	return NewInternalError(err.Error())
}

func httpErrorType(statusCode int) string {
	switch statusCode {
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "invalid_request_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	default:
		return "server_error"
	}
}

// APIError is an error in the OpenAI format.
type APIError struct {
	StatusCode int    `json:"-"`
	Type       string `json:"type"`
	Message    string `json:"message"`
	Code       string `json:"code,omitempty"`
	Param      string `json:"param,omitempty"`
}

func (e *APIError) Error() string {
	return e.Message
}

// ErrorResponse is the standard OpenAI wrapper.
type ErrorResponse struct {
	Error APIError `json:"error"`
}

func NewBadRequestError(message string) *APIError {
	return &APIError{
		StatusCode: http.StatusBadRequest,
		Type:       "invalid_request_error",
		Message:    message,
	}
}

func NewUnauthorizedError(message string) *APIError {
	return &APIError{
		StatusCode: http.StatusUnauthorized,
		Type:       "authentication_error",
		Message:    message,
	}
}

func NewRateLimitError(message string) *APIError {
	return &APIError{
		StatusCode: http.StatusTooManyRequests,
		Type:       "rate_limit_error",
		Message:    message,
	}
}

func NewInternalError(message string) *APIError {
	return &APIError{
		StatusCode: http.StatusInternalServerError,
		Type:       "server_error",
		Message:    message,
	}
}

func NewModelNotFoundError(model string) *APIError {
	return &APIError{
		StatusCode: http.StatusNotFound,
		Type:       "invalid_request_error",
		Message:    "The model `" + model + "` does not exist.",
		Code:       "model_not_found",
	}
}

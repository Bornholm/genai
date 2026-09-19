package codec

import (
	stderrors "errors"

	"github.com/bornholm/genai/llm"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/pkg/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ErrorToProto captures the parts of an error the host needs to rebuild a
// typed one: the retry decorator relies on llm.IsRetryable, callers on
// errors.Is against the llm sentinels.
func ErrorToProto(err error) *pluginv1.Error {
	if err == nil {
		return nil
	}
	result := &pluginv1.Error{Message: err.Error()}

	var httpErr *llm.HTTPError
	var validationErr llm.ValidationError
	switch {
	case stderrors.As(err, &httpErr):
		result.StatusCode = int32(httpErr.StatusCode)
		result.Body = httpErr.Body
		if stderrors.Is(err, llm.ErrRateLimit) {
			result.Kind = pluginv1.Error_KIND_RATE_LIMIT
		} else {
			result.Kind = pluginv1.Error_KIND_HTTP
		}
	case stderrors.Is(err, llm.ErrRateLimit):
		result.Kind = pluginv1.Error_KIND_RATE_LIMIT
		result.StatusCode = 429
	case stderrors.Is(err, llm.ErrNoMessage):
		result.Kind = pluginv1.Error_KIND_NO_MESSAGE
	case stderrors.Is(err, llm.ErrUnavailable):
		result.Kind = pluginv1.Error_KIND_UNAVAILABLE
	case stderrors.As(err, &validationErr):
		result.Kind = pluginv1.Error_KIND_VALIDATION
		result.Field = validationErr.Field
		result.Message = validationErr.Message
	}
	return result
}

// ErrorFromProto rebuilds a typed error. A nil proto gives a nil error.
func ErrorFromProto(e *pluginv1.Error) error {
	if e == nil {
		return nil
	}
	switch e.GetKind() {
	case pluginv1.Error_KIND_RATE_LIMIT:
		status := int(e.GetStatusCode())
		if status == 0 {
			status = 429
		}
		return errors.WithStack(llm.RateLimitError(status, e.GetBody()))
	case pluginv1.Error_KIND_HTTP:
		return errors.WithStack(llm.NewHTTPError(int(e.GetStatusCode()), e.GetBody()))
	case pluginv1.Error_KIND_NO_MESSAGE:
		return errors.WithStack(llm.ErrNoMessage)
	case pluginv1.Error_KIND_UNAVAILABLE:
		return errors.Wrap(llm.ErrUnavailable, e.GetMessage())
	case pluginv1.Error_KIND_VALIDATION:
		return llm.NewValidationError(e.GetField(), e.GetMessage())
	default:
		return errors.New(e.GetMessage())
	}
}

// ErrorToStatus wraps an error into a gRPC status carrying its typed form as
// a detail, for unary RPCs. Context errors keep their gRPC code so that the
// caller can tell a cancellation from a provider failure.
func ErrorToStatus(err error) error {
	if err == nil {
		return nil
	}
	code := codes.Unknown
	switch {
	case stderrors.Is(err, llm.ErrUnavailable):
		code = codes.Unavailable
	case stderrors.Is(err, llm.ErrRateLimit):
		code = codes.ResourceExhausted
	}
	var validationErr llm.ValidationError
	if stderrors.As(err, &validationErr) {
		code = codes.InvalidArgument
	}
	st := status.New(code, err.Error())
	withDetails, detailErr := st.WithDetails(ErrorToProto(err))
	if detailErr != nil {
		return st.Err()
	}
	return withDetails.Err()
}

// ErrorFromStatus is the inverse of ErrorToStatus. An error that is not a
// gRPC status, or carries no typed detail, is returned wrapped as is.
func ErrorFromStatus(err error) error {
	if err == nil {
		return nil
	}
	st, ok := status.FromError(err)
	if !ok {
		return errors.WithStack(err)
	}
	for _, detail := range st.Details() {
		if typed, ok := detail.(*pluginv1.Error); ok {
			return ErrorFromProto(typed)
		}
	}
	switch st.Code() {
	case codes.Unavailable:
		return errors.Wrap(llm.ErrUnavailable, st.Message())
	case codes.Canceled:
		return errors.WithStack(err)
	default:
		return errors.WithStack(err)
	}
}

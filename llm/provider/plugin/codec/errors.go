package codec

import (
	"context"
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
	case stderrors.Is(err, context.Canceled):
		result.Kind = pluginv1.Error_KIND_CANCELED
	case stderrors.Is(err, context.DeadlineExceeded):
		result.Kind = pluginv1.Error_KIND_DEADLINE_EXCEEDED
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
		return withMessage(llm.RateLimitError(status, e.GetBody()), e.GetMessage())
	case pluginv1.Error_KIND_HTTP:
		return withMessage(llm.NewHTTPError(int(e.GetStatusCode()), e.GetBody()), e.GetMessage())
	case pluginv1.Error_KIND_NO_MESSAGE:
		return withMessage(llm.ErrNoMessage, e.GetMessage())
	case pluginv1.Error_KIND_UNAVAILABLE:
		return withMessage(llm.ErrUnavailable, e.GetMessage())
	case pluginv1.Error_KIND_VALIDATION:
		return llm.NewValidationError(e.GetField(), e.GetMessage())
	case pluginv1.Error_KIND_CANCELED:
		return withMessage(context.Canceled, e.GetMessage())
	case pluginv1.Error_KIND_DEADLINE_EXCEEDED:
		return withMessage(context.DeadlineExceeded, e.GetMessage())
	default:
		return errors.New(e.GetMessage())
	}
}

// withMessage keeps the text the plugin side produced around a typed error
// (provider name, failing call) while errors.Is and errors.As still reach the
// rebuilt typed error underneath.
func withMessage(err error, message string) error {
	if message == "" || message == err.Error() {
		return errors.WithStack(err)
	}
	return errors.WithStack(&wireError{message: message, err: err})
}

// wireError is a typed error whose text is the one that crossed the wire.
type wireError struct {
	message string
	err     error
}

func (e *wireError) Error() string { return e.message }
func (e *wireError) Unwrap() error { return e.err }

// ErrorToStatus wraps an error into a gRPC status carrying its typed form as
// a detail, for unary RPCs. Context errors get their gRPC code so that the
// caller can tell a cancellation from a provider failure even without the
// detail.
func ErrorToStatus(err error) error {
	if err == nil {
		return nil
	}
	code := codes.Unknown
	switch {
	case stderrors.Is(err, context.Canceled):
		code = codes.Canceled
	case stderrors.Is(err, context.DeadlineExceeded):
		code = codes.DeadlineExceeded
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
	// No typed detail: a plugin not built with the SDK, or a failure of the
	// transport itself. Map what the gRPC code alone says.
	switch st.Code() {
	case codes.Unavailable:
		return errors.Wrap(llm.ErrUnavailable, st.Message())
	case codes.NotFound:
		// The plugin does not know the client: it restarted since it was
		// configured.
		return errors.Wrap(llm.ErrUnavailable, st.Message())
	case codes.ResourceExhausted:
		return errors.Wrap(llm.ErrRateLimit, st.Message())
	case codes.Canceled:
		// Either the host's own context (grpc-go reports it as a status) or
		// the plugin's: callers check errors.Is(err, context.Canceled).
		return withMessage(context.Canceled, st.Message())
	case codes.DeadlineExceeded:
		return withMessage(context.DeadlineExceeded, st.Message())
	default:
		return errors.WithStack(err)
	}
}

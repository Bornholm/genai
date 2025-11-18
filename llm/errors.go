package llm

import (
	"fmt"

	"github.com/pkg/errors"
)

// WrapError provides consistent error wrapping with stack traces
func WrapError(err error, msg string, args ...interface{}) error {
	if err == nil {
		return nil
	}
	return errors.Wrapf(errors.WithStack(err), msg, args...)
}

// NewError creates a new error with stack trace
func NewError(msg string, args ...interface{}) error {
	return errors.WithStack(fmt.Errorf(msg, args...))
}

// ValidationError represents a validation error
type ValidationError struct {
	Field   string
	Message string
}

func (e ValidationError) Error() string {
	return fmt.Sprintf("validation error for field '%s': %s", e.Field, e.Message)
}

// NewValidationError creates a new validation error
func NewValidationError(field, message string) error {
	return errors.WithStack(ValidationError{
		Field:   field,
		Message: message,
	})
}

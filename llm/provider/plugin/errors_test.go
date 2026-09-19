package plugin

import (
	"errors"
)

// errorsAs is errors.As without the linter complaint about a pointer-to-
// interface target in tests.
func errorsAs(err error, target any) bool {
	return errors.As(err, target)
}

package loop

import "errors"

// ErrIterationBudgetExceeded is returned when the loop exceeds its maximum iteration count
var ErrIterationBudgetExceeded = errors.New("iteration budget exceeded")

package filter

import (
	"context"

	"github.com/bornholm/genai/llm"
)

// FilterRule is checked against every incoming message list.
// Returning a non-nil error blocks the request.
type FilterRule interface {
	Check(ctx context.Context, messages []llm.Message) error
}

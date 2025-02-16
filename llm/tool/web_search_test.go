package tool

import (
	"context"
	"testing"
	"time"

	"github.com/pkg/errors"
)

func TestWebSearch(t *testing.T) {
	search := WebSearch()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := search.Execute(ctx, map[string]any{"topic": "generative artificial intelligence"})
	if err != nil {
		t.Fatalf("%+v", errors.WithStack(err))
	}

	t.Logf("Result:\n\n%s", result)
}

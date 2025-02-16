package index

import (
	"context"
	"testing"
	"time"

	"github.com/pkg/errors"
)

func TestIndex(t *testing.T) {
	search, err := Search(
		WithResourceCollections(
			WebsiteCollection("https://www.cadoles.com"),
		),
	)
	if err != nil {
		t.Fatalf("%+v", errors.WithStack(err))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := search.Execute(ctx, map[string]any{"topic": "EOLE"})
	if err != nil {
		t.Fatalf("%+v", errors.WithStack(err))
	}

	t.Logf("Result:\n\n%s", result)
}

package tool

import (
	"context"
	"testing"
	"time"

	"github.com/pkg/errors"
)

func TestBrowse(t *testing.T) {
	browse := Browse()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := browse.Execute(ctx, map[string]any{"url": "https://linuxfr.org"})
	if err != nil {
		t.Fatalf("%+v", errors.WithStack(err))
	}

	t.Logf("Result:\n\n%s", result)
}

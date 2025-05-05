package marker

import (
	"context"
	"net/url"

	"github.com/bornholm/genai/extract"
	"github.com/bornholm/genai/extract/provider"
)

const Name provider.Name = "marker"

func init() {
	provider.RegisterTextClient(Name, func(ctx context.Context, dsn *url.URL) (extract.TextClient, error) {
		dsn.Scheme = "http"
		if dsn.Query().Has("useTLS") {
			dsn.Scheme = "https"
		}

		return NewTextClient(dsn), nil
	})
}

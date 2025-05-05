package mistral

import (
	"context"
	"net/url"

	"github.com/bornholm/genai/extract"
	"github.com/bornholm/genai/extract/provider"
)

const Name provider.Name = "mistral"

func init() {
	provider.RegisterTextClient(Name, func(ctx context.Context, dsn *url.URL) (extract.TextClient, error) {
		dsn.Scheme = "http"
		if dsn.Query().Has("useTLS") {
			dsn.Scheme = "https"
		}

		query := dsn.Query()

		apiKey := query.Get("apiKey")

		dsn.RawQuery = query.Encode()

		return NewTextClient(dsn, apiKey), nil
	})
}

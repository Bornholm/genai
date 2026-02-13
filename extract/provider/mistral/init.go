package mistral

import (
	"context"
	"net/url"

	"github.com/bornholm/genai/extract"
	"github.com/bornholm/genai/extract/provider"
	"github.com/pkg/errors"
)

const mistralAPIBaseURL = "https://api.mistral.ai"

const Name provider.Name = "mistral"

func init() {
	provider.RegisterTextClient(Name, func(ctx context.Context, dsn *url.URL) (extract.TextClient, error) {
		query := dsn.Query()

		apiKey := query.Get("apiKey")
		query.Del("apiKey")
		dsn.RawQuery = query.Encode()

		baseURL, err := url.Parse(mistralAPIBaseURL)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		return NewTextClient(baseURL, apiKey), nil
	})
}

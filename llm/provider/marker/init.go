package marker

import (
	"context"
	"net/url"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/pkg/errors"
)

const Name provider.Name = "marker"

func init() {
	provider.RegisterExtractText(Name, func(ctx context.Context, opts provider.ClientOptions) (llm.ExtractTextClient, error) {
		baseURL, err := url.Parse(opts.BaseURL)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		return NewExtractTextClient(baseURL), nil
	})
}

package provider

import (
	"context"

	"github.com/bornholm/genai/extract"
)

var defaultRegistry = NewRegistry()

func RegisterTextClient(name Name, factory Factory[extract.TextClient]) {
	defaultRegistry.RegisterTextClient(name, factory)
}

func Create(ctx context.Context, funcs ...OptionFunc) (extract.Client, Name, error) {
	return defaultRegistry.Create(ctx, funcs...)
}

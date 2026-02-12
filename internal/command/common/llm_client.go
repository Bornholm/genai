package common

import (
	"context"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/circuitbreaker"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/env"
	"github.com/bornholm/genai/llm/ratelimit"
	"github.com/bornholm/genai/llm/retry"
	"github.com/pkg/errors"
	"golang.org/x/time/rate"
)

func NewResilientClient(ctx context.Context, envPrefix string, envFile string) (llm.Client, error) {
	var (
		client llm.Client
		err    error
	)

	client, err = provider.Create(ctx, env.With(envPrefix, envFile))
	if err != nil {
		return nil, errors.WithStack(err)
	}

	client = retry.Wrap(client, time.Second, 5)
	client = ratelimit.Wrap(client, time.Duration(rate.Every(time.Second)), 1)
	client = circuitbreaker.NewClient(client, 5, 30*time.Second)

	return client, nil
}

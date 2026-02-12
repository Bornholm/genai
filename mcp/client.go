package mcp

import (
	"context"

	"github.com/bornholm/genai/llm"
)

type Client interface {
	Start(ctx context.Context) error
	Stop() error

	GetTools(ctx context.Context) ([]llm.Tool, error)
}

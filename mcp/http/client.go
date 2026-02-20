package http

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/mcp"
	"github.com/bornholm/genai/mcp/common"
	"github.com/pkg/errors"

	goMCP "github.com/modelcontextprotocol/go-sdk/mcp"
)

type Client struct {
	client *common.Client
}

// GetTools implements [mcp.Client].
func (c *Client) GetTools(ctx context.Context) ([]llm.Tool, error) {
	return c.client.GetTools(ctx)
}

// Start implements [mcp.Client].
func (c *Client) Start(ctx context.Context) error {
	return c.client.Start(ctx)
}

// Stop implements [mcp.Client].
func (c *Client) Stop() error {
	return c.client.Stop()
}

func NewClient(endpoint string, funcs ...OptionFunc) *Client {
	opts := NewOptions(funcs...)

	var connector common.ConnectorFunc = func(ctx context.Context) (*goMCP.ClientSession, error) {
		client := goMCP.NewClient(&goMCP.Implementation{Name: "mcp-client", Version: "v1.0.0"}, nil)

		transport := &goMCP.SSEClientTransport{
			Endpoint:   endpoint,
			HTTPClient: opts.HTTPClient,
		}

		session, err := client.Connect(ctx, transport, nil)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		return session, nil
	}

	return &Client{
		client: common.NewClient(connector),
	}
}

var _ mcp.Client = &Client{}

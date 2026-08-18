// Package streamable implements the MCP "Streamable HTTP" transport
// (specification revision 2025-03-26 and later), where every message is a POST
// to a single endpoint and the server may answer with a stream.
//
// It is distinct from package mcp/http, which implements the older HTTP+SSE
// transport (revision 2024-11-05). Servers rarely speak both: a Streamable
// HTTP server typically rejects the standalone GET the SSE client opens first.
package streamable

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

		transport := &goMCP.StreamableClientTransport{
			Endpoint:             endpoint,
			HTTPClient:           opts.HTTPClient,
			MaxRetries:           opts.MaxRetries,
			DisableStandaloneSSE: opts.DisableStandaloneSSE,
		}

		session, err := client.Connect(ctx, transport, nil)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		return session, nil
	}

	return &Client{
		client: common.NewClient(connector,
			common.WithMaxRetries(opts.MaxRetries),
			common.WithBaseDelay(opts.BaseDelay),
			common.WithReconnectEnabled(opts.ReconnectEnabled),
			common.WithReadOnlyHint(opts.ReadOnlyHint)),
	}
}

var _ mcp.Client = &Client{}

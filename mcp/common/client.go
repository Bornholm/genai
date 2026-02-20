package common

import (
	"context"
	"strings"
	"sync"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/mcp"
	"github.com/go-viper/mapstructure/v2"
	goMCP "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/pkg/errors"
)

type Connector interface {
	Connect(ctx context.Context) (*goMCP.ClientSession, error)
}

type ConnectorFunc func(ctx context.Context) (*goMCP.ClientSession, error)

func (fn ConnectorFunc) Connect(ctx context.Context) (*goMCP.ClientSession, error) {
	return fn(ctx)
}

type Client struct {
	connector Connector

	mutex sync.RWMutex

	session *goMCP.ClientSession
}

// GetTools implements [mcp.Client].
func (c *Client) GetTools(ctx context.Context) ([]llm.Tool, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if c.session == nil {
		return nil, errors.New("mcp session not initialized")
	}

	tools := make([]llm.Tool, 0)

	cursor := ""

	for {
		res, err := c.session.ListTools(ctx, &goMCP.ListToolsParams{Cursor: cursor})
		if err != nil {
			return nil, errors.WithStack(err)
		}

		if len(res.Tools) == 0 {
			break
		}

		for _, t := range res.Tools {
			llmTool, err := c.toTool(t)
			if err != nil {
				return nil, errors.WithStack(err)
			}

			tools = append(tools, llmTool)
		}

		if res.NextCursor == "" {
			break
		}

		cursor = res.NextCursor
	}

	return tools, nil
}

func (c *Client) toTool(t *goMCP.Tool) (llm.Tool, error) {
	var parameters map[string]any

	if err := mapstructure.Decode(t.InputSchema, &parameters); err != nil {
		return nil, errors.Wrapf(err, "could not initialize tool '%s'", t.Name)
	}

	return llm.NewFuncTool(
		t.Name,
		t.Description,
		parameters,
		func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			c.mutex.RLock()
			defer c.mutex.RUnlock()

			res, err := c.session.CallTool(ctx, &goMCP.CallToolParams{
				Name:      t.Name,
				Arguments: params,
			})
			if err != nil {
				return nil, errors.WithStack(err)
			}

			var sb strings.Builder

			if res.IsError {
				sb.WriteString("ERROR:\n")
			}

			for _, c := range res.Content {
				textContent, ok := c.(*goMCP.TextContent)
				if ok {
					sb.WriteString(textContent.Text)
				}
			}

			return llm.NewToolResult(sb.String()), nil
		},
	), nil
}

// Start implements [mcp.Client].
func (c *Client) Start(ctx context.Context) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	session, err := c.connector.Connect(ctx)
	if err != nil {
		return errors.WithStack(err)
	}

	c.session = session

	return nil
}

// Stop implements [mcp.Client].
func (c *Client) Stop() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	var err error

	if c.session != nil {
		err = c.session.Close()
	}

	c.session = nil

	return err
}

func NewClient(connector Connector) *Client {
	return &Client{
		connector: connector,
	}
}

var _ mcp.Client = &Client{}

package common

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"time"

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

type Options struct {
	MaxRetries       int
	BaseDelay        time.Duration
	ReconnectEnabled bool
}

type Client struct {
	connector Connector
	options   Options

	mutex sync.RWMutex

	session *goMCP.ClientSession
}

func (c *Client) reconnect(ctx context.Context) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.session != nil {
		_ = c.session.Close()
		c.session = nil
	}

	session, err := c.connector.Connect(ctx)
	if err != nil {
		return errors.WithStack(err)
	}

	c.session = session
	return nil
}

func isConnectionClosed(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	return strings.Contains(errStr, "connection closed") ||
		strings.Contains(errStr, "client is closing") ||
		strings.Contains(errStr, "eof") ||
		strings.Contains(errStr, "use of closed network connection")
}

func (c *Client) callToolWithReconnect(ctx context.Context, toolName string, params map[string]any) (*goMCP.CallToolResult, error) {
	maxRetries := c.options.MaxRetries
	baseDelay := c.options.BaseDelay
	backoff := baseDelay

	for attempt := 0; attempt <= maxRetries; attempt++ {
		res, err := c.doCallTool(ctx, toolName, params)
		if err == nil {
			return res, nil
		}

		if !isConnectionClosed(err) {
			return nil, errors.WithStack(err)
		}

		if attempt < maxRetries {
			slog.DebugContext(ctx, "mcp connection closed, attempting reconnect",
				slog.String("tool", toolName),
				slog.Int("attempt", attempt+1),
				slog.Duration("backoff", backoff))

			if reconnectErr := c.reconnect(ctx); reconnectErr != nil {
				slog.WarnContext(ctx, "mcp reconnect failed",
					slog.String("tool", toolName),
					slog.Any("error", reconnectErr))
			} else {
				slog.DebugContext(ctx, "mcp reconnected successfully",
					slog.String("tool", toolName))
				time.Sleep(backoff)
				backoff *= 2
				continue
			}
		}

		time.Sleep(backoff)
		backoff *= 2
	}

	return nil, errors.Errorf("failed to execute tool '%s' after %d retries", toolName, maxRetries)
}

func (c *Client) doCallTool(ctx context.Context, toolName string, params map[string]any) (*goMCP.CallToolResult, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if c.session == nil {
		return nil, errors.New("mcp session not initialized")
	}

	res, err := c.session.CallTool(ctx, &goMCP.CallToolParams{
		Name:      toolName,
		Arguments: params,
	})
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return res, nil
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
			res, err := c.callToolWithReconnect(ctx, t.Name, params)
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

func NewClient(connector Connector, opts ...OptionFunc) *Client {
	options := Options{
		MaxRetries:       3,
		BaseDelay:        100 * time.Millisecond,
		ReconnectEnabled: true,
	}
	for _, fn := range opts {
		fn(&options)
	}

	return &Client{
		connector: connector,
		options:   options,
	}
}

type OptionFunc func(opts *Options)

func WithMaxRetries(maxRetries int) OptionFunc {
	return func(opts *Options) {
		opts.MaxRetries = maxRetries
	}
}

func WithBaseDelay(delay time.Duration) OptionFunc {
	return func(opts *Options) {
		opts.BaseDelay = delay
	}
}

func WithReconnectEnabled(enabled bool) OptionFunc {
	return func(opts *Options) {
		opts.ReconnectEnabled = enabled
	}
}

var _ mcp.Client = &Client{}

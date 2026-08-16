package common

import (
	"context"
	"encoding/base64"
	"fmt"
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
	ReadOnlyHint     bool
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
			if c.options.ReadOnlyHint && (t.Annotations == nil || !t.Annotations.ReadOnlyHint) {
				continue
			}

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

	tool := llm.NewFuncTool(
		t.Name,
		t.Description,
		parameters,
		func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
			res, err := c.callToolWithReconnect(ctx, t.Name, params)
			if err != nil {
				return nil, errors.WithStack(err)
			}

			return toToolResult(ctx, res), nil
		},
	)

	// Carry the readOnlyHint annotation through to the caller, who may use it
	// to decide whether a tool needs confirmation before running.
	//
	// Only set it when the server actually sent annotations. ReadOnlyHint is
	// a plain bool with omitempty in the protocol types, so an absent
	// annotation and an explicit "this tool writes" are indistinguishable
	// once Annotations exists; a nil Annotations is the one reliable signal
	// that the server declared nothing at all.
	if t.Annotations != nil {
		tool = tool.WithReadOnlyHint(t.Annotations.ReadOnlyHint)
	}

	return tool, nil
}

// toToolResult turns the content blocks of a tool call into a [llm.ToolResult].
//
// A tool may answer with media as well as text — an image, a sound, a binary
// resource — and dropping those blocks makes the tool look empty: the model is
// told nothing came back and reports to the user that it cannot see the image
// it just asked for. Media therefore become attachments, and each one also
// leaves a textual marker so a caller that cannot forward attachments (the
// "tool" role carries text only for most providers) still states that a medium
// was returned instead of pretending there was none.
func toToolResult(ctx context.Context, res *goMCP.CallToolResult) llm.ToolResult {
	var (
		sb          strings.Builder
		attachments []llm.Attachment
	)

	if res.IsError {
		sb.WriteString("ERROR:\n")
	}

	appendAttachment := func(kind llm.AttachmentType, mimeType string, data []byte) {
		attachment, err := llm.NewBase64Attachment(kind, mimeType, base64.StdEncoding.EncodeToString(data))
		if err != nil {
			// Best-effort: a medium the attachment layer refuses must not fail
			// the tool call, whose text may well be the useful part.
			slog.WarnContext(ctx, "could not convert mcp content to attachment",
				slog.String("mimeType", mimeType), slog.Any("error", errors.WithStack(err)))

			return
		}

		attachments = append(attachments, attachment)
		fmt.Fprintf(&sb, "\n[%s attachment: %s, %d bytes]\n", kind, mimeType, len(data))
	}

	for _, content := range res.Content {
		switch content := content.(type) {
		case *goMCP.TextContent:
			sb.WriteString(content.Text)

		case *goMCP.ImageContent:
			appendAttachment(llm.AttachmentTypeImage, content.MIMEType, content.Data)

		case *goMCP.AudioContent:
			appendAttachment(llm.AttachmentTypeAudio, content.MIMEType, content.Data)

		case *goMCP.EmbeddedResource:
			if content.Resource == nil {
				continue
			}

			// A resource carries either text or bytes, never both.
			if content.Resource.Blob != nil {
				appendAttachment(attachmentTypeOf(content.Resource.MIMEType), content.Resource.MIMEType, content.Resource.Blob)

				continue
			}

			sb.WriteString(content.Resource.Text)

		case *goMCP.ResourceLink:
			// A link is a reference to fetch, not content: name it so the model
			// can decide to read it.
			fmt.Fprintf(&sb, "\n[resource link: %s]\n", content.URI)
		}
	}

	return llm.NewToolResult(sb.String(), attachments...)
}

// attachmentTypeOf maps a media type onto the attachment kind carrying it,
// falling back on a document — the generic kind — for anything else.
func attachmentTypeOf(mimeType string) llm.AttachmentType {
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		return llm.AttachmentTypeImage
	case strings.HasPrefix(mimeType, "audio/"):
		return llm.AttachmentTypeAudio
	case strings.HasPrefix(mimeType, "video/"):
		return llm.AttachmentTypeVideo
	default:
		return llm.AttachmentTypeDocument
	}
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

func WithReadOnlyHint(enabled bool) OptionFunc {
	return func(opts *Options) {
		opts.ReadOnlyHint = enabled
	}
}

var _ mcp.Client = &Client{}

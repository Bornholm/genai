package common

import (
	"context"
	"log/slog"
	"net/url"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/mcp"
	"github.com/bornholm/genai/mcp/http"
	"github.com/bornholm/genai/mcp/stdio"
	"github.com/bornholm/go-x/slogx"
	"github.com/google/shlex"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"
)

func GetMCPTools(ctx *cli.Context, mcpURLParam string, mcpAuthParam string) ([]llm.Tool, func(), error) {
	mcpURLs := ctx.StringSlice(mcpURLParam)
	mcpAuths := ctx.StringSlice(mcpAuthParam)

	return GetMCPToolsFromURLs(ctx.Context, mcpURLs, mcpAuths)
}

func GetMCPToolsFromURLs(ctx context.Context, mcpURLs []string, mcpAuths []string) ([]llm.Tool, func(), error) {
	clients := make([]mcp.Client, 0)

	close := func() {
		for _, c := range clients {
			if err := c.Stop(); err != nil {
				slog.ErrorContext(ctx, "could not stop mcp client", slogx.Error(err))
			}
		}
	}

	for i, u := range mcpURLs {
		var c mcp.Client
		if parsedURL, err := url.ParseRequestURI(u); err == nil && strings.HasPrefix(parsedURL.Scheme, "http") {
			opts := []http.OptionFunc{}

			if len(mcpAuths) > i {
				opts = append(opts, http.WithAuthToken(mcpAuths[i]))
			}

			c = http.NewClient(u, opts...)
		} else {
			command, err := shlex.Split(u)
			if err != nil {
				return nil, nil, errors.Wrapf(err, "could not parse mcp server command '%s'", u)
			}
			c = stdio.NewClient(command)
		}

		slog.DebugContext(ctx, "starting mcp client", slog.String("client", u))

		if err := c.Start(ctx); err != nil {
			return nil, nil, errors.Wrapf(err, "could not start mcp client '%s'", u)
		}

		slog.DebugContext(ctx, "mcp client started", slog.String("client", u))

		clients = append(clients, c)
	}

	tools := make([]llm.Tool, 0)

	for i, c := range clients {
		clientTools, err := c.GetTools(ctx)
		if err != nil {
			return nil, nil, errors.Wrapf(err, "could not retrieve mcp server '%s' tools", mcpURLs[i])
		}

		tools = append(tools, clientTools...)
	}

	return tools, close, nil
}

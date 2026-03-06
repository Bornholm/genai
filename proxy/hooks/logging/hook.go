package logging

import (
	"context"
	"log/slog"
	"time"

	"github.com/bornholm/genai/proxy"
)

// requestStartKey is used to store the request start time in Metadata.
const requestStartKey = "_logging.start"

// Hook logs incoming requests and outgoing responses using a configurable
// slog.Logger. It implements PreRequestHook and PostResponseHook.
//
// Log levels:
//   - info:  one line per request (type, model, user) and one on response
//     (status, duration, tokens used).
//   - debug: additionally logs raw metadata attached by other hooks.
type Hook struct {
	logger   *slog.Logger
	priority int
}

// Name implements proxy.Hook.
func (h *Hook) Name() string { return "logging" }

// Priority implements proxy.Hook.
func (h *Hook) Priority() int { return h.priority }

// PreRequest implements proxy.PreRequestHook.
func (h *Hook) PreRequest(ctx context.Context, req *proxy.ProxyRequest) (*proxy.HookResult, error) {
	req.Metadata[requestStartKey] = time.Now()

	h.logger.InfoContext(ctx, "proxy request received",
		slog.String("type", string(req.Type)),
		slog.String("model", req.Model),
		slog.String("user", req.UserID),
	)

	h.logger.DebugContext(ctx, "proxy request detail",
		slog.Any("metadata", req.Metadata),
		slog.Int("body_bytes", len(req.Body)),
	)

	return nil, nil
}

// PostResponse implements proxy.PostResponseHook.
func (h *Hook) PostResponse(ctx context.Context, req *proxy.ProxyRequest, res *proxy.ProxyResponse) (*proxy.HookResult, error) {
	elapsed := elapsedSince(req.Metadata)

	attrs := []any{
		slog.String("type", string(req.Type)),
		slog.String("model", req.Model),
		slog.String("user", req.UserID),
		slog.Int("status", res.StatusCode),
		slog.Duration("duration", elapsed),
	}

	if res.TokensUsed != nil {
		attrs = append(attrs,
			slog.Int("prompt_tokens", res.TokensUsed.PromptTokens),
			slog.Int("completion_tokens", res.TokensUsed.CompletionTokens),
			slog.Int("total_tokens", res.TokensUsed.TotalTokens),
		)
	}

	h.logger.InfoContext(ctx, "proxy request completed", attrs...)

	h.logger.DebugContext(ctx, "proxy response detail",
		slog.String("model", req.Model),
		slog.Any("metadata", req.Metadata),
	)

	return nil, nil
}

func elapsedSince(meta map[string]any) time.Duration {
	v, ok := meta[requestStartKey]
	if !ok {
		return 0
	}
	start, ok := v.(time.Time)
	if !ok {
		return 0
	}
	return time.Since(start)
}

// New creates a Hook using logger at the given priority.
// Pass nil to use the default slog logger.
func New(logger *slog.Logger, priority int) *Hook {
	if logger == nil {
		logger = slog.Default()
	}
	return &Hook{logger: logger, priority: priority}
}

var _ proxy.PreRequestHook = &Hook{}
var _ proxy.PostResponseHook = &Hook{}

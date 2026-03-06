package logging

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"strings"
	"testing"

	"github.com/bornholm/genai/proxy"
)

func newTestLogger(buf *bytes.Buffer, level slog.Level) *slog.Logger {
	return slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: level}))
}

func TestHook_LogsRequestAndResponse(t *testing.T) {
	var buf bytes.Buffer
	logger := newTestLogger(&buf, slog.LevelDebug)
	h := New(logger, 0)

	req := &proxy.ProxyRequest{
		Type:     proxy.RequestTypeChatCompletion,
		Model:    "gpt-4",
		UserID:   "alice",
		Metadata: map[string]any{},
	}

	// PreRequest should log and record start time
	result, err := h.PreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("PreRequest error: %v", err)
	}
	if result != nil {
		t.Error("PreRequest should not short-circuit")
	}

	if _, ok := req.Metadata[requestStartKey]; !ok {
		t.Error("start time not recorded in metadata")
	}

	preLog := buf.String()
	if !strings.Contains(preLog, "proxy request received") {
		t.Errorf("expected 'proxy request received' in log, got: %s", preLog)
	}
	if !strings.Contains(preLog, "gpt-4") {
		t.Errorf("expected model name in log, got: %s", preLog)
	}
	if !strings.Contains(preLog, "alice") {
		t.Errorf("expected user ID in log, got: %s", preLog)
	}

	buf.Reset()

	// PostResponse should log with duration and token info
	res := &proxy.ProxyResponse{
		StatusCode: http.StatusOK,
		TokensUsed: &proxy.TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}

	hookResult, err := h.PostResponse(context.Background(), req, res)
	if err != nil {
		t.Fatalf("PostResponse error: %v", err)
	}
	if hookResult != nil {
		t.Error("PostResponse should not return a result")
	}

	postLog := buf.String()
	if !strings.Contains(postLog, "proxy request completed") {
		t.Errorf("expected 'proxy request completed' in log, got: %s", postLog)
	}
	if !strings.Contains(postLog, "total_tokens=15") {
		t.Errorf("expected token info in log, got: %s", postLog)
	}
	if !strings.Contains(postLog, "duration=") {
		t.Errorf("expected duration in log, got: %s", postLog)
	}
}

func TestHook_InfoLevelSuppressesDebug(t *testing.T) {
	var buf bytes.Buffer
	logger := newTestLogger(&buf, slog.LevelInfo)
	h := New(logger, 0)

	req := &proxy.ProxyRequest{
		Type:     proxy.RequestTypeChatCompletion,
		Model:    "gpt-4",
		UserID:   "bob",
		Body:     []byte(`{"model":"gpt-4"}`),
		Metadata: map[string]any{},
	}

	_, _ = h.PreRequest(context.Background(), req)

	log := buf.String()
	if strings.Contains(log, "proxy request detail") {
		t.Error("debug messages should be suppressed at info level")
	}
	if !strings.Contains(log, "proxy request received") {
		t.Error("info message should still appear")
	}
}

func TestHook_NilLogger_UsesDefault(t *testing.T) {
	h := New(nil, 0)
	if h.logger == nil {
		t.Error("logger should not be nil when nil is passed")
	}
}

func TestHook_NoTokens_DoesNotPanic(t *testing.T) {
	var buf bytes.Buffer
	h := New(newTestLogger(&buf, slog.LevelInfo), 0)

	req := &proxy.ProxyRequest{
		Type:     proxy.RequestTypeEmbedding,
		Model:    "ada",
		Metadata: map[string]any{},
	}
	_, _ = h.PreRequest(context.Background(), req)

	res := &proxy.ProxyResponse{StatusCode: http.StatusOK} // no TokensUsed
	if _, err := h.PostResponse(context.Background(), req, res); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

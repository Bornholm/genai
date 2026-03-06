package filter

import (
	"context"
	"net/http"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/proxy"
)

func chatReq(msgs ...llm.Message) *proxy.ProxyRequest {
	opts := []llm.ChatCompletionOptionFunc{llm.WithMessages(msgs...)}
	return &proxy.ProxyRequest{
		Type:        proxy.RequestTypeChatCompletion,
		ChatOptions: opts,
		Metadata:    map[string]any{},
	}
}

// ---- KeywordRule --------------------------------------------------------

func TestKeywordRule_Block(t *testing.T) {
	rule := NewKeywordRule("badword")
	msgs := []llm.Message{llm.NewMessage(llm.RoleUser, "Please say badword now")}
	if err := rule.Check(context.Background(), msgs); err == nil {
		t.Error("expected error for blocked keyword")
	}
}

func TestKeywordRule_Allow(t *testing.T) {
	rule := NewKeywordRule("badword")
	msgs := []llm.Message{llm.NewMessage(llm.RoleUser, "Hello world")}
	if err := rule.Check(context.Background(), msgs); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestKeywordRule_CaseInsensitive(t *testing.T) {
	rule := NewKeywordRule("BadWord")
	msgs := []llm.Message{llm.NewMessage(llm.RoleUser, "BADWORD BADWORD")}
	if err := rule.Check(context.Background(), msgs); err == nil {
		t.Error("expected block for case-insensitive match")
	}
}

// ---- RegexRule ----------------------------------------------------------

func TestRegexRule_Block(t *testing.T) {
	rule, err := NewRegexRule(`\bcredit card\b`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msgs := []llm.Message{llm.NewMessage(llm.RoleUser, "my credit card number is 1234")}
	if err := rule.Check(context.Background(), msgs); err == nil {
		t.Error("expected block")
	}
}

func TestRegexRule_Allow(t *testing.T) {
	rule, _ := NewRegexRule(`\bcredit card\b`)
	msgs := []llm.Message{llm.NewMessage(llm.RoleUser, "Hello")}
	if err := rule.Check(context.Background(), msgs); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRegexRule_InvalidPattern(t *testing.T) {
	_, err := NewRegexRule(`[invalid`)
	if err == nil {
		t.Error("expected error for invalid regex")
	}
}

// ---- MaxTokenRule -------------------------------------------------------

func TestMaxTokenRule_Block(t *testing.T) {
	rule := NewMaxTokenRule(10)
	msgs := []llm.Message{llm.NewMessage(llm.RoleUser, "This is a very long message that exceeds the limit")}
	if err := rule.Check(context.Background(), msgs); err == nil {
		t.Error("expected block for oversized message")
	}
}

func TestMaxTokenRule_Allow(t *testing.T) {
	rule := NewMaxTokenRule(100)
	msgs := []llm.Message{llm.NewMessage(llm.RoleUser, "short")}
	if err := rule.Check(context.Background(), msgs); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// ---- ContentFilter ------------------------------------------------------

func TestContentFilter_BlocksRequest(t *testing.T) {
	f := NewContentFilter(1, NewKeywordRule("forbidden"))
	req := chatReq(llm.NewMessage(llm.RoleUser, "this is forbidden content"))

	result, err := f.PreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil || result.Response == nil {
		t.Fatal("expected blocked response")
	}
	if result.Response.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", result.Response.StatusCode, http.StatusBadRequest)
	}
}

func TestContentFilter_AllowsRequest(t *testing.T) {
	f := NewContentFilter(1, NewKeywordRule("forbidden"))
	req := chatReq(llm.NewMessage(llm.RoleUser, "hello world"))

	result, err := f.PreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil && result.Response != nil {
		t.Error("should not have blocked request")
	}
}

func TestContentFilter_IgnoresNonChatRequest(t *testing.T) {
	f := NewContentFilter(1, NewKeywordRule("forbidden"))
	req := &proxy.ProxyRequest{
		Type:     proxy.RequestTypeEmbedding,
		Metadata: map[string]any{},
	}

	result, err := f.PreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("embedding request should not be filtered")
	}
}

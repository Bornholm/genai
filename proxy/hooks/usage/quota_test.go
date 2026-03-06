package usage

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/bornholm/genai/proxy"
)

func seedStore(t *testing.T, store UsageStore, userID string, records []UsageRecord) {
	t.Helper()
	for _, r := range records {
		if err := store.Record(context.Background(), r); err != nil {
			t.Fatalf("seed record: %v", err)
		}
	}
}

func todayRecord(userID string, prompt, completion int) UsageRecord {
	return UsageRecord{
		UserID:           userID,
		Model:            "gpt-4",
		PromptTokens:     prompt,
		CompletionTokens: completion,
		Timestamp:        time.Now(),
		RequestType:      proxy.RequestTypeChatCompletion,
	}
}

func TestQuotaEnforcer_TokenQuota_Exceeded(t *testing.T) {
	store := NewInMemoryUsageStore()
	seedStore(t, store, "user1", []UsageRecord{
		todayRecord("user1", 800, 200), // 1000 tokens total
	})

	enforcer := NewQuotaEnforcer(store, map[string]QuotaConfig{
		"*": {MaxTokensPerDay: 999},
	}, 0)

	req := &proxy.ProxyRequest{UserID: "user1", Model: "gpt-4", Metadata: map[string]any{}}
	result, err := enforcer.PreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil || result.Response == nil {
		t.Fatal("expected quota exceeded response")
	}
	if result.Response.StatusCode != http.StatusTooManyRequests {
		t.Errorf("status = %d, want %d", result.Response.StatusCode, http.StatusTooManyRequests)
	}
}

func TestQuotaEnforcer_TokenQuota_NotExceeded(t *testing.T) {
	store := NewInMemoryUsageStore()
	seedStore(t, store, "user1", []UsageRecord{
		todayRecord("user1", 100, 50), // 150 tokens
	})

	enforcer := NewQuotaEnforcer(store, map[string]QuotaConfig{
		"*": {MaxTokensPerDay: 10000},
	}, 0)

	req := &proxy.ProxyRequest{UserID: "user1", Model: "gpt-4", Metadata: map[string]any{}}
	result, err := enforcer.PreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil && result.Response != nil {
		t.Error("should not have blocked request")
	}
}

func TestQuotaEnforcer_RequestQuota_Exceeded(t *testing.T) {
	store := NewInMemoryUsageStore()
	for i := 0; i < 5; i++ {
		seedStore(t, store, "user2", []UsageRecord{todayRecord("user2", 10, 5)})
	}

	enforcer := NewQuotaEnforcer(store, map[string]QuotaConfig{
		"*": {MaxRequestsPerDay: 4},
	}, 0)

	req := &proxy.ProxyRequest{UserID: "user2", Model: "gpt-4", Metadata: map[string]any{}}
	result, err := enforcer.PreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil || result.Response == nil {
		t.Fatal("expected quota exceeded response")
	}
	if result.Response.StatusCode != http.StatusTooManyRequests {
		t.Errorf("status = %d, want %d", result.Response.StatusCode, http.StatusTooManyRequests)
	}
}

func TestQuotaEnforcer_NoQuotaConfigured(t *testing.T) {
	store := NewInMemoryUsageStore()
	enforcer := NewQuotaEnforcer(store, map[string]QuotaConfig{}, 0)

	req := &proxy.ProxyRequest{UserID: "user3", Model: "gpt-4", Metadata: map[string]any{}}
	result, err := enforcer.PreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil && result.Response != nil {
		t.Error("should not block when no quota configured")
	}
}

func TestQuotaEnforcer_UserSpecificOverridesDefault(t *testing.T) {
	store := NewInMemoryUsageStore()
	seedStore(t, store, "vip", []UsageRecord{
		todayRecord("vip", 5000, 5000), // 10000 tokens
	})

	enforcer := NewQuotaEnforcer(store, map[string]QuotaConfig{
		"*":   {MaxTokensPerDay: 1000},  // default: 1000
		"vip": {MaxTokensPerDay: 50000}, // vip: 50000
	}, 0)

	req := &proxy.ProxyRequest{UserID: "vip", Model: "gpt-4", Metadata: map[string]any{}}
	result, err := enforcer.PreRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil && result.Response != nil {
		t.Error("vip user should not be blocked")
	}
}

// ---- InMemoryUsageStore -------------------------------------------------

func TestInMemoryUsageStore_Record(t *testing.T) {
	store := NewInMemoryUsageStore()
	r := todayRecord("u1", 10, 5)
	if err := store.Record(context.Background(), r); err != nil {
		t.Fatalf("record: %v", err)
	}

	records, err := store.GetUsage(context.Background(), "u1", time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatalf("get usage: %v", err)
	}
	if len(records) != 1 {
		t.Errorf("records = %d, want 1", len(records))
	}
}

func TestInMemoryUsageStore_GetTotalTokens(t *testing.T) {
	store := NewInMemoryUsageStore()
	seedStore(t, store, "u2", []UsageRecord{
		todayRecord("u2", 100, 50),
		todayRecord("u2", 200, 100),
	})

	total, err := store.GetTotalTokens(context.Background(), "u2", time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatalf("get total: %v", err)
	}
	if total != 450 {
		t.Errorf("total = %d, want 450", total)
	}
}

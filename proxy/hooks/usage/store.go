package usage

import (
	"context"
	"sync"
	"time"

	"github.com/bornholm/genai/proxy"
	"github.com/pkg/errors"
)

// UsageRecord captures a single LLM call's metrics.
type UsageRecord struct {
	UserID           string
	Model            string
	PromptTokens     int
	CompletionTokens int
	Timestamp        time.Time
	RequestType      proxy.RequestType
}

// UsageStore persists and queries usage records.
type UsageStore interface {
	Record(ctx context.Context, record UsageRecord) error
	GetUsage(ctx context.Context, userID string, since time.Time) ([]UsageRecord, error)
	GetTotalTokens(ctx context.Context, userID string, since time.Time) (int, error)
	GetTotalRequests(ctx context.Context, userID string, since time.Time) (int, error)
}

// InMemoryUsageStore is a thread-safe in-process store, useful for development
// and testing.
type InMemoryUsageStore struct {
	mu      sync.RWMutex
	records []UsageRecord
}

// Record implements UsageStore.
func (s *InMemoryUsageStore) Record(ctx context.Context, record UsageRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.records = append(s.records, record)
	return nil
}

// GetUsage implements UsageStore.
func (s *InMemoryUsageStore) GetUsage(ctx context.Context, userID string, since time.Time) ([]UsageRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var out []UsageRecord
	for _, r := range s.records {
		if r.UserID == userID && !r.Timestamp.Before(since) {
			out = append(out, r)
		}
	}
	return out, nil
}

// GetTotalTokens implements UsageStore.
func (s *InMemoryUsageStore) GetTotalTokens(ctx context.Context, userID string, since time.Time) (int, error) {
	records, err := s.GetUsage(ctx, userID, since)
	if err != nil {
		return 0, errors.WithStack(err)
	}
	total := 0
	for _, r := range records {
		total += r.PromptTokens + r.CompletionTokens
	}
	return total, nil
}

// GetTotalRequests implements UsageStore.
func (s *InMemoryUsageStore) GetTotalRequests(ctx context.Context, userID string, since time.Time) (int, error) {
	records, err := s.GetUsage(ctx, userID, since)
	if err != nil {
		return 0, errors.WithStack(err)
	}
	return len(records), nil
}

// NewInMemoryUsageStore creates an empty in-memory store.
func NewInMemoryUsageStore() *InMemoryUsageStore {
	return &InMemoryUsageStore{}
}

var _ UsageStore = &InMemoryUsageStore{}

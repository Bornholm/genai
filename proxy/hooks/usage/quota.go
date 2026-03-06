package usage

import (
	"context"
	"net/http"
	"time"

	"github.com/bornholm/genai/proxy"
)

// QuotaConfig defines rate limits for a user (or the default "*").
type QuotaConfig struct {
	MaxTokensPerDay   int
	MaxRequestsPerDay int
	// ModelLimits overrides the top-level limits for specific model names.
	ModelLimits map[string]QuotaConfig
}

// QuotaEnforcer is a PreRequestHook that blocks requests when a user exceeds
// their configured daily quota.
type QuotaEnforcer struct {
	store    UsageStore
	quotas   map[string]QuotaConfig // userID → config; "*" = default
	priority int
}

// Name implements proxy.Hook.
func (q *QuotaEnforcer) Name() string { return "usage.quota" }

// Priority implements proxy.Hook.
func (q *QuotaEnforcer) Priority() int { return q.priority }

// PreRequest implements proxy.PreRequestHook.
func (q *QuotaEnforcer) PreRequest(ctx context.Context, req *proxy.ProxyRequest) (*proxy.HookResult, error) {
	cfg, ok := q.resolveConfig(req.UserID, req.Model)
	if !ok {
		return nil, nil
	}

	since := startOfDay()

	if cfg.MaxRequestsPerDay > 0 {
		count, err := q.store.GetTotalRequests(ctx, req.UserID, since)
		if err == nil && count >= cfg.MaxRequestsPerDay {
			apiErr := proxy.NewRateLimitError("daily request quota exceeded")
			return &proxy.HookResult{
				Response: &proxy.ProxyResponse{
					StatusCode: http.StatusTooManyRequests,
					Body:       proxy.ErrorResponse{Error: *apiErr},
				},
			}, nil
		}
	}

	if cfg.MaxTokensPerDay > 0 {
		tokens, err := q.store.GetTotalTokens(ctx, req.UserID, since)
		if err == nil && tokens >= cfg.MaxTokensPerDay {
			apiErr := proxy.NewRateLimitError("daily token quota exceeded")
			return &proxy.HookResult{
				Response: &proxy.ProxyResponse{
					StatusCode: http.StatusTooManyRequests,
					Body:       proxy.ErrorResponse{Error: *apiErr},
				},
			}, nil
		}
	}

	return nil, nil
}

// resolveConfig returns the most specific quota config for the given user/model pair.
func (q *QuotaEnforcer) resolveConfig(userID, model string) (QuotaConfig, bool) {
	// Try user-specific config first
	if cfg, ok := q.quotas[userID]; ok {
		if model != "" {
			if modelCfg, ok := cfg.ModelLimits[model]; ok {
				return modelCfg, true
			}
		}
		return cfg, true
	}
	// Fall back to default
	if cfg, ok := q.quotas["*"]; ok {
		if model != "" {
			if modelCfg, ok := cfg.ModelLimits[model]; ok {
				return modelCfg, true
			}
		}
		return cfg, true
	}
	return QuotaConfig{}, false
}

func startOfDay() time.Time {
	now := time.Now()
	return time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
}

// NewQuotaEnforcer creates a QuotaEnforcer.
// Use "*" as a userID key in quotas to set the default quota for all users.
func NewQuotaEnforcer(store UsageStore, quotas map[string]QuotaConfig, priority int) *QuotaEnforcer {
	return &QuotaEnforcer{
		store:    store,
		quotas:   quotas,
		priority: priority,
	}
}

var _ proxy.PreRequestHook = &QuotaEnforcer{}

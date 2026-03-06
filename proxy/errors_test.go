package proxy

import (
	"net/http"
	"testing"
)

func TestErrorConstructors(t *testing.T) {
	tests := []struct {
		name       string
		err        *APIError
		wantStatus int
		wantType   string
	}{
		{
			name:       "bad request",
			err:        NewBadRequestError("bad"),
			wantStatus: http.StatusBadRequest,
			wantType:   "invalid_request_error",
		},
		{
			name:       "unauthorized",
			err:        NewUnauthorizedError("auth failed"),
			wantStatus: http.StatusUnauthorized,
			wantType:   "authentication_error",
		},
		{
			name:       "rate limit",
			err:        NewRateLimitError("quota"),
			wantStatus: http.StatusTooManyRequests,
			wantType:   "rate_limit_error",
		},
		{
			name:       "internal",
			err:        NewInternalError("boom"),
			wantStatus: http.StatusInternalServerError,
			wantType:   "server_error",
		},
		{
			name:       "model not found",
			err:        NewModelNotFoundError("gpt-99"),
			wantStatus: http.StatusNotFound,
			wantType:   "invalid_request_error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.err.StatusCode != tt.wantStatus {
				t.Errorf("StatusCode = %d, want %d", tt.err.StatusCode, tt.wantStatus)
			}
			if tt.err.Type != tt.wantType {
				t.Errorf("Type = %q, want %q", tt.err.Type, tt.wantType)
			}
			if tt.err.Message == "" {
				t.Error("Message must not be empty")
			}
			if tt.err.Error() == "" {
				t.Error("Error() must not return empty string")
			}
		})
	}
}

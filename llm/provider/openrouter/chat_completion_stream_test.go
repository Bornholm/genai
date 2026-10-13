package openrouter

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/revrost/go-openrouter"
)

// newTestClient returns a client whose requests reach handler.
func newTestClient(t *testing.T, handler http.HandlerFunc) *ChatCompletionClient {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	config := openrouter.DefaultConfig("test-key")
	config.BaseURL = server.URL
	return NewChatCompletionClient(openrouter.NewClientWithConfig(*config), "test/model")
}

// sse answers with the given SSE events, as OpenRouter streams them.
func sse(events ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = w.Write([]byte("data: " + event + "\n\n"))
		}
	}
}

// collect reads the stream to its end and returns its content and last chunk.
func collect(t *testing.T, client *ChatCompletionClient) (string, llm.StreamChunk) {
	t.Helper()
	stream, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "hi")))
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}
	var content strings.Builder
	var last llm.StreamChunk
	for chunk := range stream {
		if chunk.Delta() != nil {
			content.WriteString(chunk.Delta().Content())
		}
		last = chunk
	}
	return content.String(), last
}

const (
	helloChunk  = `{"choices":[{"index":0,"delta":{"content":"Hello"}}]}`
	finishChunk = `{"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}`
	usageChunk  = `{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`
)

func TestChatCompletionStream_CompletesAFinishedAnswer(t *testing.T) {
	client := newTestClient(t, sse(helloChunk, finishChunk, usageChunk, "[DONE]"))

	content, last := collect(t, client)

	if content != "Hello!" {
		t.Errorf("content = %q, want %q", content, "Hello!")
	}
	if last == nil || !last.IsComplete() || last.Error() != nil {
		t.Fatalf("last chunk = %v, want a completion chunk", last)
	}
	if got := last.Usage().TotalTokens(); got != 7 {
		t.Errorf("total tokens = %d, want 7", got)
	}
}

func TestChatCompletionStream_ReportsATruncatedStream(t *testing.T) {
	for name, handler := range map[string]http.HandlerFunc{
		"connection closed": sse(helloChunk),
		"undecodable line":  sse(helloChunk, `{"choices":`),
	} {
		t.Run(name, func(t *testing.T) {
			content, last := collect(t, newTestClient(t, handler))

			if content != "Hello" {
				t.Errorf("content = %q, want what was streamed before the cut", content)
			}
			if last == nil || last.Error() == nil {
				t.Fatalf("last chunk = %v, want an error: the answer is incomplete", last)
			}
			if !llm.IsRetryable(last.Error()) {
				t.Errorf("error = %v, want a retryable error", last.Error())
			}
		})
	}
}

func TestChatCompletionStream_ReportsAProviderFailure(t *testing.T) {
	failure := `{"error":{"code":502,"message":"provider disconnected"},"choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}`
	client := newTestClient(t, sse(helloChunk, failure, "[DONE]"))

	_, last := collect(t, client)

	if last == nil || !isProviderFailure(last.Error()) {
		t.Fatalf("last chunk = %v, want the provider failure", last)
	}
}

func TestChatCompletionStream_MapsAnAPIErrorToAnHTTPError(t *testing.T) {
	for _, tc := range []struct {
		status    int
		retryable bool
	}{
		{http.StatusTooManyRequests, true},
		{http.StatusBadGateway, true},
		{http.StatusBadRequest, false},
	} {
		t.Run(http.StatusText(tc.status), func(t *testing.T) {
			client := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(`{"error":{"code":1,"message":"refused"}}`))
			})

			_, last := collect(t, client)

			var httpErr *llm.HTTPError
			if last == nil || !errors.As(last.Error(), &httpErr) || httpErr.StatusCode != tc.status {
				t.Fatalf("last chunk = %v, want an *llm.HTTPError with status %d", last, tc.status)
			}
			if got := llm.IsRetryable(last.Error()); got != tc.retryable {
				t.Errorf("IsRetryable = %v, want %v", got, tc.retryable)
			}
		})
	}
}

func TestChatCompletion_MapsAnAPIErrorToAnHTTPError(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":429,"message":"slow down"}}`))
	})

	_, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "hi")))

	if !errors.Is(err, llm.ErrRateLimit) || !llm.IsRetryable(err) {
		t.Errorf("err = %v, want a retryable rate limit error", err)
	}
}

// TestChatCompletionStream_CompletesWhenTheContextEndsAfterTheAnswer asserts
// that a finished answer is reported complete although the caller's context
// ended before the stream closed: the answer was delivered in full.
func TestChatCompletionStream_CompletesWhenTheContextEndsAfterTheAnswer(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []string{helloChunk, finishChunk} {
			_, _ = w.Write([]byte("data: " + event + "\n\n"))
		}
		w.(http.Flusher).Flush()
		// Hold the stream open, without [DONE], until the caller leaves.
		<-r.Context().Done()
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stream, err := client.ChatCompletionStream(ctx, llm.WithMessages(llm.NewMessage(llm.RoleUser, "hi")))
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}
	var last llm.StreamChunk
	for chunk := range stream {
		if chunk.Delta() != nil && chunk.Delta().Content() == "!" {
			cancel()
		}
		last = chunk
	}

	if last == nil || !last.IsComplete() || last.Error() != nil {
		t.Fatalf("last chunk = %v, want a completion chunk", last)
	}
}

func TestChatCompletionStream_ReportsAFailureGivenAsNativeFinishReason(t *testing.T) {
	failure := `{"choices":[{"index":0,"delta":{"content":""},"native_finish_reason":"error"}]}`
	client := newTestClient(t, sse(helloChunk, failure, "[DONE]"))

	_, last := collect(t, client)

	if last == nil || !isProviderFailure(last.Error()) {
		t.Fatalf("last chunk = %v, want the provider failure", last)
	}
}

// TestChatCompletionStream_ClosesTheUpstreamOnAProviderFailure asserts that
// a stream stopped on a provider failure closes its upstream response, even
// when the caller's context never ends and OpenRouter keeps it open.
func TestChatCompletionStream_ClosesTheUpstreamOnAProviderFailure(t *testing.T) {
	failure := `{"choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}`
	closed := make(chan struct{})
	testDone := make(chan struct{})
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []string{helloChunk, failure, helloChunk} {
			_, _ = w.Write([]byte("data: " + event + "\n\n"))
		}
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
			close(closed)
		case <-testDone:
			// Lets the server shut down when the response was left open.
		}
	})
	// Cleanups run last in, first out: this one runs before the server's.
	t.Cleanup(func() { close(testDone) })

	_, last := collect(t, client)
	if last == nil || !isProviderFailure(last.Error()) {
		t.Fatalf("last chunk = %v, want the provider failure", last)
	}

	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("the upstream response is still open after the stream ended")
	}
}

func TestEmbeddings_MapsAnUnparsableErrorToAnHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("<html>upstream unavailable</html>"))
	}))
	t.Cleanup(server.Close)
	config := openrouter.DefaultConfig("test-key")
	config.BaseURL = server.URL

	_, err := NewEmbeddingsClient(openrouter.NewClientWithConfig(*config), "test/model").Embeddings(context.Background(), []string{"hi"})

	var httpErr *llm.HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusServiceUnavailable || !llm.IsRetryable(err) {
		t.Errorf("err = %v, want a retryable *llm.HTTPError with status 503", err)
	}
}

// isProviderFailure reports whether err is the error of an answer the provider
// failed to produce.
func isProviderFailure(err error) bool {
	var httpErr *llm.HTTPError
	return errors.As(err, &httpErr) && httpErr.Body == errProviderFailed().(*llm.HTTPError).Body
}

func TestChatCompletionStream_ReportsAFailureWhateverItsCase(t *testing.T) {
	failure := `{"choices":[{"index":0,"delta":{"content":""},"native_finish_reason":"ERROR"}]}`
	client := newTestClient(t, sse(helloChunk, failure, "[DONE]"))

	_, last := collect(t, client)

	if last == nil || !isProviderFailure(last.Error()) {
		t.Fatalf("last chunk = %v, want the provider failure", last)
	}
}

func TestChatCompletion_ReportsAProviderFailure(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"Hel"},"finish_reason":"error"}]}`))
	})

	_, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "hi")))

	if !isProviderFailure(err) || !llm.IsRetryable(err) {
		t.Errorf("err = %v, want a retryable provider failure", err)
	}
}

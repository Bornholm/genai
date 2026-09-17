package proxy

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"syscall"
	"testing"

	"github.com/bornholm/genai/llm"
)

// brokenPipeWriter fails every write past the first one with syscall.EPIPE,
// mimicking a client that hangs up mid-stream.
type brokenPipeWriter struct {
	header http.Header
	writes int
	code   int
}

func (w *brokenPipeWriter) Header() http.Header {
	if w.header == nil {
		w.header = http.Header{}
	}
	return w.header
}

func (w *brokenPipeWriter) WriteHeader(code int) { w.code = code }

func (w *brokenPipeWriter) Write(p []byte) (int, error) {
	w.writes++
	if w.writes > 1 {
		return 0, syscall.EPIPE
	}
	return len(p), nil
}

func (w *brokenPipeWriter) Flush() {}

// countingStreamClient emits chunks deltas and records how many were consumed.
type countingStreamClient struct {
	chunks int

	mu       sync.Mutex
	produced int
}

func (c *countingStreamClient) producedCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.produced
}

func (c *countingStreamClient) ChatCompletionStream(ctx context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	out := make(chan llm.StreamChunk)
	go func() {
		defer close(out)
		for i := 0; i < c.chunks; i++ {
			chunk := llm.StreamChunk(llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "tok")))
			select {
			case out <- chunk:
				c.mu.Lock()
				c.produced++
				c.mu.Unlock()
			case <-ctx.Done():
				return
			}
		}
		select {
		case out <- llm.StreamChunk(llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(5, 3, 8))):
		case <-ctx.Done():
		}
	}()
	return out, nil
}

func (c *countingStreamClient) ChatCompletion(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return nil, nil
}

func (c *countingStreamClient) Embeddings(_ context.Context, _ []string, _ ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

func (c *countingStreamClient) Transcription(_ context.Context, _ []byte, _ ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	return nil, nil
}

// TestStreamChatCompletion_StopsOnClientDisconnect asserts that a failed write
// ends the stream instead of draining the rest of the upstream chunks, each of
// them logging an error against a connection nobody reads.
func TestStreamChatCompletion_StopsOnClientDisconnect(t *testing.T) {
	client := &countingStreamClient{chunks: 50}
	resolver := &resolverHook{client: client, model: "gpt-4"}
	postCalled := false
	post := &stubPostHook{hookName: "usage", hookPriority: 1, called: &postCalled}

	server := NewServer(WithHook(resolver), WithHook(post))

	w := &brokenPipeWriter{}
	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	// One successful write for the first chunk, one failed write for the next:
	// nothing more should be attempted.
	if w.writes != 2 {
		t.Errorf("writes = %d, want 2 (the stream kept emitting after the client went away)", w.writes)
	}
	if produced := client.producedCount(); produced > 3 {
		t.Errorf("consumed %d upstream chunks after the client went away, want at most 3", produced)
	}
	if !postCalled {
		t.Error("post-response hook was not run: usage consumed before the disconnect goes unrecorded")
	}
}

func TestIsClientGone(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()

	for _, tc := range []struct {
		name string
		ctx  context.Context
		err  error
		want bool
	}{
		{"nil error", context.Background(), nil, false},
		{"broken pipe", context.Background(), syscall.EPIPE, true},
		{"connection reset", context.Background(), syscall.ECONNRESET, true},
		{"wrapped broken pipe", context.Background(), errors.Join(errors.New("write tcp"), syscall.EPIPE), true},
		{"canceled context", canceled, errors.New("some write error"), true},
		{"genuine failure", context.Background(), errors.New("marshal failed"), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := isClientGone(tc.ctx, tc.err); got != tc.want {
				t.Errorf("isClientGone() = %v, want %v", got, tc.want)
			}
		})
	}
}

var _ http.ResponseWriter = &brokenPipeWriter{}
var _ http.Flusher = &brokenPipeWriter{}

// erroringStreamClient emits okChunks good deltas and then a chunk carrying an
// error, mimicking a provider that fails after the response started flowing.
type erroringStreamClient struct {
	okChunks int
	err      error
}

func (c *erroringStreamClient) ChatCompletionStream(ctx context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	out := make(chan llm.StreamChunk)
	go func() {
		defer close(out)
		for i := 0; i < c.okChunks; i++ {
			chunk := llm.StreamChunk(llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "tok")))
			select {
			case out <- chunk:
			case <-ctx.Done():
				return
			}
		}
		select {
		case out <- llm.StreamChunk(llm.NewErrorStreamChunk(c.err)):
		case <-ctx.Done():
		}
	}()
	return out, nil
}

func (c *erroringStreamClient) ChatCompletion(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return nil, nil
}

func (c *erroringStreamClient) Embeddings(_ context.Context, _ []string, _ ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

func (c *erroringStreamClient) Transcription(_ context.Context, _ []byte, _ ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	return nil, nil
}

// capturingPostHook keeps the response it was handed, so a test can assert on
// the usage and the interruption the chain reported.
type capturingPostHook struct {
	res *ProxyResponse
}

func (h *capturingPostHook) Name() string  { return "test.capture" }
func (h *capturingPostHook) Priority() int { return 1 }
func (h *capturingPostHook) PostResponse(_ context.Context, _ *ProxyRequest, res *ProxyResponse) (*HookResult, error) {
	h.res = res
	return nil, nil
}

// TestStreamChatCompletion_RecordsUsageOnUpstreamError asserts that a provider
// failing mid-stream still runs the post-response hooks with the partial usage.
// The tokens produced before the failure reached the client and were billed by
// the provider, so skipping the hooks loses them for usage, quotas and costs.
func TestStreamChatCompletion_RecordsUsageOnUpstreamError(t *testing.T) {
	client := &erroringStreamClient{okChunks: 3, err: errors.New("upstream exploded")}
	resolver := &resolverHook{client: client, model: "gpt-4"}
	capture := &capturingPostHook{}

	server := NewServer(WithHook(resolver), WithHook(capture))

	w := httptest.NewRecorder()
	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	if capture.res == nil {
		t.Fatal("post-response hook was not run: usage produced before the upstream error goes unrecorded")
	}
	if capture.res.Interruption == nil {
		t.Fatal("Interruption is nil: hooks cannot tell the stream was cut short")
	}
	if got, want := capture.res.Interruption.Cause, StreamInterruptionUpstream; got != want {
		t.Errorf("Interruption.Cause = %q, want %q", got, want)
	}
	if capture.res.Interruption.Err == nil {
		t.Error("Interruption.Err is nil, want the upstream error")
	}
	if got := capture.res.Interruption.ChunksEmitted; got != 3 {
		t.Errorf("Interruption.ChunksEmitted = %d, want 3", got)
	}
	if capture.res.TokensUsed == nil {
		t.Fatal("TokensUsed is nil, want the partial counts collected before the error")
	}
}

// TestStreamChatCompletion_ReportsClientHangup asserts that the other way a
// stream stops early is reported with its own cause, so accounting can tell an
// upstream failure from a client that went away.
func TestStreamChatCompletion_ReportsClientHangup(t *testing.T) {
	client := &countingStreamClient{chunks: 50}
	resolver := &resolverHook{client: client, model: "gpt-4"}
	capture := &capturingPostHook{}

	server := NewServer(WithHook(resolver), WithHook(capture))

	w := &brokenPipeWriter{}
	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	if capture.res == nil {
		t.Fatal("post-response hook was not run")
	}
	if capture.res.Interruption == nil {
		t.Fatal("Interruption is nil, want a client hangup")
	}
	if got, want := capture.res.Interruption.Cause, StreamInterruptionClientGone; got != want {
		t.Errorf("Interruption.Cause = %q, want %q", got, want)
	}
}

// TestStreamChatCompletion_NoInterruptionOnCompletedStream guards the normal
// path: a stream that runs to completion reports no interruption at all.
func TestStreamChatCompletion_NoInterruptionOnCompletedStream(t *testing.T) {
	client := &countingStreamClient{chunks: 3}
	resolver := &resolverHook{client: client, model: "gpt-4"}
	capture := &capturingPostHook{}

	server := NewServer(WithHook(resolver), WithHook(capture))

	w := httptest.NewRecorder()
	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	if capture.res == nil {
		t.Fatal("post-response hook was not run")
	}
	if capture.res.Interruption != nil {
		t.Errorf("Interruption = %+v, want nil on a completed stream", capture.res.Interruption)
	}
}

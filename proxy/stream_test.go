package proxy

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/bornholm/genai/llm"
)

// brokenPipeWriter fails every write past failAfter with syscall.EPIPE,
// mimicking a client that hangs up mid-stream. A zero failAfter means the very
// first write succeeds and nothing else does.
type brokenPipeWriter struct {
	header    http.Header
	writes    int
	code      int
	failAfter int
	body      bytes.Buffer
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
	limit := w.failAfter
	if limit == 0 {
		limit = 1
	}
	if w.writes > limit {
		return 0, syscall.EPIPE
	}
	w.body.Write(p)
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
// Each delta carries the usage published so far — what Anthropic does, and what
// makes the partial counts of an interrupted stream more than zeroes. With
// silentUsage set it publishes nothing instead, the way the OpenAI-compatible
// providers behave, which is the case where the counts stay unknown.
type erroringStreamClient struct {
	okChunks     int
	err          error
	silentUsage  bool
	promptTokens int64
}

func (c *erroringStreamClient) ChatCompletionStream(ctx context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	out := make(chan llm.StreamChunk)
	go func() {
		defer close(out)
		for i := 0; i < c.okChunks; i++ {
			delta := llm.NewStreamDelta(llm.RoleAssistant, "tok")
			var chunk llm.StreamChunk = llm.NewStreamChunk(delta)
			if !c.silentUsage {
				// Cumulative, one completion token per delta emitted.
				completion := int64(i + 1)
				chunk = llm.NewStreamChunkWithUsage(delta, llm.NewChatCompletionUsage(
					c.promptTokens, completion, c.promptTokens+completion,
				))
			}
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
	client := &erroringStreamClient{okChunks: 3, promptTokens: 7, err: errors.New("upstream exploded")}
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
	if capture.res.Interruption.ErrorEventUndelivered {
		t.Error("ErrorEventUndelivered is true, but the client read the error event")
	}
	if !capture.res.Interruption.PartialUsage {
		t.Error("PartialUsage is false, but the provider published its counters before failing")
	}
	if capture.res.TokensUsed == nil {
		t.Fatal("TokensUsed is nil, want the partial counts collected before the error")
	}
	// Asserting the values, not just non-nilness: TokensUsed is always
	// allocated, so a nil check passes just as well on an empty row.
	if got, want := capture.res.TokensUsed.PromptTokens, 7; got != want {
		t.Errorf("TokensUsed.PromptTokens = %d, want %d", got, want)
	}
	if got, want := capture.res.TokensUsed.CompletionTokens, 3; got != want {
		t.Errorf("TokensUsed.CompletionTokens = %d, want %d", got, want)
	}

	// The client must see the error and never the normal end of stream: a
	// "[DONE]" here would let it treat a truncated answer as a whole one.
	body := w.Body.String()
	if !strings.Contains(body, "upstream exploded") {
		t.Errorf("response body does not carry the error event:\n%s", body)
	}
	if strings.Contains(body, "[DONE]") {
		t.Errorf("response body ends with [DONE] although the stream was cut short:\n%s", body)
	}
}

// TestStreamChatCompletion_ReportsUnknownUsageOnUpstreamError guards the honest
// half of the contract: with a provider that only reports usage in its final
// chunk — every OpenAI-compatible one — an interruption leaves the counts
// unknown. PartialUsage says so, so that a zeroed TokensUsed is not billed as a
// free request.
func TestStreamChatCompletion_ReportsUnknownUsageOnUpstreamError(t *testing.T) {
	client := &erroringStreamClient{okChunks: 3, silentUsage: true, err: errors.New("upstream exploded")}
	resolver := &resolverHook{client: client, model: "gpt-4"}
	capture := &capturingPostHook{}

	server := NewServer(WithHook(resolver), WithHook(capture))

	w := httptest.NewRecorder()
	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	if capture.res == nil || capture.res.Interruption == nil {
		t.Fatal("no interruption reported")
	}
	if capture.res.Interruption.PartialUsage {
		t.Error("PartialUsage is true although the provider never published any usage")
	}
	if got := capture.res.TokensUsed.TotalTokens; got != 0 {
		t.Errorf("TokensUsed.TotalTokens = %d, want 0 when nothing was published", got)
	}
	if got := capture.res.Interruption.ChunksEmitted; got != 3 {
		t.Errorf("Interruption.ChunksEmitted = %d, want 3 as the volume proxy", got)
	}
}

// TestStreamChatCompletion_ReportsUndeliveredErrorEvent covers the double
// failure: the provider dies mid-stream and the client is not there to read the
// error event either. The cause stays the upstream failure — that is what
// stopped the stream — but the flag keeps a hook from assuming the client was
// told.
func TestStreamChatCompletion_ReportsUndeliveredErrorEvent(t *testing.T) {
	client := &erroringStreamClient{okChunks: 3, promptTokens: 7, err: errors.New("upstream exploded")}
	resolver := &resolverHook{client: client, model: "gpt-4"}
	capture := &capturingPostHook{}

	server := NewServer(WithHook(resolver), WithHook(capture))

	// The three deltas go through, the error event that follows does not.
	w := &brokenPipeWriter{failAfter: 3}
	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	if capture.res == nil || capture.res.Interruption == nil {
		t.Fatal("no interruption reported")
	}
	if got, want := capture.res.Interruption.Cause, StreamInterruptionUpstream; got != want {
		t.Errorf("Interruption.Cause = %q, want %q", got, want)
	}
	if !capture.res.Interruption.ErrorEventUndelivered {
		t.Error("ErrorEventUndelivered is false although writing the error event failed")
	}
}

// truncatingStreamClient closes its channel after a few deltas without ever
// sending a completion or an error chunk, the way a provider whose connection
// drops can leave a stream hanging.
type truncatingStreamClient struct {
	chunks int
}

func (c *truncatingStreamClient) ChatCompletionStream(ctx context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	out := make(chan llm.StreamChunk)
	go func() {
		defer close(out)
		for i := 0; i < c.chunks; i++ {
			select {
			case out <- llm.StreamChunk(llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "tok"))):
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

func (c *truncatingStreamClient) ChatCompletion(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return nil, nil
}

func (c *truncatingStreamClient) Embeddings(_ context.Context, _ []string, _ ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

func (c *truncatingStreamClient) Transcription(_ context.Context, _ []byte, _ ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	return nil, nil
}

// TestStreamChatCompletion_ReportsTruncatedStream asserts that a stream ending
// without any terminal chunk is reported as an interruption instead of being
// finalized as a complete response.
func TestStreamChatCompletion_ReportsTruncatedStream(t *testing.T) {
	client := &truncatingStreamClient{chunks: 3}
	resolver := &resolverHook{client: client, model: "gpt-4"}
	capture := &capturingPostHook{}

	server := NewServer(WithHook(resolver), WithHook(capture))

	w := httptest.NewRecorder()
	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	if capture.res == nil || capture.res.Interruption == nil {
		t.Fatal("Interruption is nil: a stream that never completed was reported as a whole response")
	}
	if got, want := capture.res.Interruption.Cause, StreamInterruptionTruncated; got != want {
		t.Errorf("Interruption.Cause = %q, want %q", got, want)
	}
	if got := capture.res.Interruption.ChunksEmitted; got != 3 {
		t.Errorf("Interruption.ChunksEmitted = %d, want 3", got)
	}
	if body := w.Body.String(); strings.Contains(body, "[DONE]") {
		t.Errorf("response body ends with [DONE] although the stream was truncated:\n%s", body)
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
	// Only the first chunk made it to the client, the write that followed is
	// what ended the stream.
	if got := capture.res.Interruption.ChunksEmitted; got != 1 {
		t.Errorf("Interruption.ChunksEmitted = %d, want 1", got)
	}
	// countingStreamClient publishes usage in its completion chunk only, which
	// a hangup never reaches: the counts are unknown, not null.
	if capture.res.Interruption.PartialUsage {
		t.Error("PartialUsage is true although no chunk carried usage")
	}
	if body := w.body.String(); strings.Contains(body, "[DONE]") {
		t.Errorf("response body ends with [DONE] although the client had gone:\n%s", body)
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

// stubbornStreamClient sends without ever watching its context, the way a
// provider whose channel writes are plain sends behaves. Dropping the channel
// on such a client strands its goroutine unless the consumer drains it.
type stubbornStreamClient struct {
	chunks int
	closed chan struct{}
}

func (c *stubbornStreamClient) ChatCompletionStream(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	out := make(chan llm.StreamChunk)
	c.closed = make(chan struct{})
	go func() {
		defer close(c.closed)
		defer close(out)
		for i := 0; i < c.chunks; i++ {
			out <- llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "tok"))
		}
		out <- llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(5, 3, 8))
	}()
	return out, nil
}

func (c *stubbornStreamClient) ChatCompletion(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return nil, nil
}

func (c *stubbornStreamClient) Embeddings(_ context.Context, _ []string, _ ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

func (c *stubbornStreamClient) Transcription(_ context.Context, _ []byte, _ ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	return nil, nil
}

// TestStreamChatCompletion_ReleasesUpstreamOnHangup asserts that abandoning an
// upstream stream also frees it. A provider goroutine left blocked on a channel
// nobody reads never runs its deferred cleanup, so the upstream HTTP response
// stays open — and billed — long after the client went away.
func TestStreamChatCompletion_ReleasesUpstreamOnHangup(t *testing.T) {
	client := &stubbornStreamClient{chunks: 50}
	resolver := &resolverHook{client: client, model: "gpt-4"}
	capture := &capturingPostHook{}

	server := NewServer(WithHook(resolver), WithHook(capture))

	w := &brokenPipeWriter{}
	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	server.handleChatCompletions(w, buildChatRequest(t, reqBody))

	select {
	case <-client.closed:
	case <-time.After(5 * time.Second):
		t.Fatal("the upstream goroutine is still blocked on its channel after the client hung up")
	}
}

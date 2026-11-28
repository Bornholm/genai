package proxy

import (
	"context"
	"errors"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/bornholm/genai/llm"
)

// delayedStreamClient announces the role at once, then waits before streaming
// its answer, the way a provider thinking before its first token does.
type delayedStreamClient struct {
	delay time.Duration
}

func (c *delayedStreamClient) ChatCompletionStream(ctx context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	out := make(chan llm.StreamChunk)
	go func() {
		defer close(out)
		chunks := []llm.StreamChunk{
			llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "")),
			llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "Hel")),
			llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "lo")),
			llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(5, 2, 7)),
		}
		for i, chunk := range chunks {
			if i == 1 {
				time.Sleep(c.delay)
			}
			select {
			case out <- chunk:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

func (c *delayedStreamClient) ChatCompletion(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return nil, nil
}

func (c *delayedStreamClient) Embeddings(_ context.Context, _ []string, _ ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

func (c *delayedStreamClient) Transcription(_ context.Context, _ []byte, _ ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	return nil, nil
}

// requestCapture keeps the request the post-response and error hooks were
// handed, so a test can read the timing the proxy recorded on it.
type requestCapture struct {
	post    *ProxyRequest
	onError *ProxyRequest
}

func (h *requestCapture) Name() string  { return "test.request-capture" }
func (h *requestCapture) Priority() int { return 1 }

func (h *requestCapture) PostResponse(_ context.Context, req *ProxyRequest, _ *ProxyResponse) (*HookResult, error) {
	copied := *req
	h.post = &copied
	return nil, nil
}

func (h *requestCapture) OnError(_ context.Context, req *ProxyRequest, _ error) (*HookResult, error) {
	copied := *req
	h.onError = &copied
	return nil, nil
}

// failingResolver refuses every model, the way an access rule does.
type failingResolver struct{}

func (failingResolver) Name() string  { return "test.failing-resolver" }
func (failingResolver) Priority() int { return 0 }
func (failingResolver) ResolveModel(context.Context, *ProxyRequest) (llm.Client, string, error) {
	return nil, "", errors.New("model refused")
}

func assertOrdered(t *testing.T, names []string, times ...time.Time) {
	t.Helper()
	for i, at := range times {
		if at.IsZero() {
			t.Fatalf("%s is zero, want it set", names[i])
		}
		if i > 0 && at.Before(times[i-1]) {
			t.Errorf("%s (%s) is before %s (%s)", names[i], at, names[i-1], times[i-1])
		}
	}
}

// TestTiming_StreamedChatCompletion asserts that a streamed exchange records
// every stage, and that the first output is the first chunk carrying text, not
// the one announcing the role: counting that one would report a provider that
// thinks for seconds as answering at once.
func TestTiming_StreamedChatCompletion(t *testing.T) {
	const delay = 30 * time.Millisecond
	capture := &requestCapture{}
	server := NewServer(
		WithHook(&resolverHook{client: &delayedStreamClient{delay: delay}, model: "gpt-4"}),
		WithHook(capture),
	)

	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildChatRequest(t, `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`))

	if capture.post == nil {
		t.Fatal("post-response hook was not run")
	}
	if !capture.post.Stream {
		t.Error("Stream = false on a streamed request")
	}
	timing := capture.post.Timing
	assertOrdered(t,
		[]string{"ReceivedAt", "UpstreamStartAt", "FirstOutputAt", "CompletedAt"},
		timing.ReceivedAt, timing.UpstreamStartAt, timing.FirstOutputAt, timing.CompletedAt,
	)

	ttfo, ok := timing.TimeToFirstOutput()
	if !ok {
		t.Fatal("TimeToFirstOutput reports no value on a stream that produced output")
	}
	if ttfo < delay {
		t.Errorf("TimeToFirstOutput = %s, want at least %s: the role-only chunk was counted as output", ttfo, delay)
	}
	if _, ok := timing.GenerationDuration(); !ok {
		t.Error("GenerationDuration reports no value on a completed stream")
	}
	if _, ok := timing.RoutingDuration(); !ok {
		t.Error("RoutingDuration reports no value")
	}
}

// TestTiming_NonStreamedChatCompletion asserts that a response that was not
// streamed has no first output, rather than one equal to its completion.
func TestTiming_NonStreamedChatCompletion(t *testing.T) {
	client := &mockChatClient{response: llm.NewChatCompletionResponse(
		llm.NewMessage(llm.RoleAssistant, "Hello!"), llm.NewChatCompletionUsage(5, 3, 8),
	)}
	capture := &requestCapture{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "gpt-4"}), WithHook(capture))

	w := httptest.NewRecorder()
	server.handleChatCompletions(w, buildChatRequest(t, `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))

	if capture.post == nil {
		t.Fatal("post-response hook was not run")
	}
	if capture.post.Stream {
		t.Error("Stream = true on a request that did not ask for it")
	}
	timing := capture.post.Timing
	assertOrdered(t,
		[]string{"ReceivedAt", "UpstreamStartAt", "CompletedAt"},
		timing.ReceivedAt, timing.UpstreamStartAt, timing.CompletedAt,
	)
	if !timing.FirstOutputAt.IsZero() {
		t.Errorf("FirstOutputAt = %s, want zero on a response that was not streamed", timing.FirstOutputAt)
	}
	if _, ok := timing.TimeToFirstOutput(); ok {
		t.Error("TimeToFirstOutput reports a value on a response that was not streamed")
	}
}

// TestTiming_ErrorHooksSeeTheFailedCall asserts that the error hooks receive
// the timing of a call the provider failed, which is all a gateway has to
// record how long a failed request took.
func TestTiming_ErrorHooksSeeTheFailedCall(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "completion"
		body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
		var client llm.Client = &mockChatClient{err: errors.New("provider down")}
		if stream {
			name = "stream"
			body = `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
			client = &erroringStreamClient{err: errors.New("provider down")}
		}
		t.Run(name, func(t *testing.T) {
			capture := &requestCapture{}
			server := NewServer(WithHook(&resolverHook{client: client, model: "gpt-4"}), WithHook(capture))

			server.handleChatCompletions(httptest.NewRecorder(), buildChatRequest(t, body))

			if capture.onError == nil {
				t.Fatal("error hook was not run")
			}
			timing := capture.onError.Timing
			assertOrdered(t,
				[]string{"ReceivedAt", "UpstreamStartAt", "CompletedAt"},
				timing.ReceivedAt, timing.UpstreamStartAt, timing.CompletedAt,
			)
			if !timing.FirstOutputAt.IsZero() {
				t.Errorf("FirstOutputAt = %s, want zero: the call failed before any output", timing.FirstOutputAt)
			}
		})
	}
}

// TestTiming_UpstreamNeverCalled asserts that a request refused before the
// upstream call reports no upstream stage, rather than a zero-length one.
func TestTiming_UpstreamNeverCalled(t *testing.T) {
	capture := &requestCapture{}
	server := NewServer(WithHook(failingResolver{}), WithHook(capture))

	server.handleChatCompletions(httptest.NewRecorder(), buildChatRequest(t, `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))

	if capture.onError == nil {
		t.Fatal("error hook was not run")
	}
	timing := capture.onError.Timing
	if timing.ReceivedAt.IsZero() {
		t.Error("ReceivedAt is zero")
	}
	if !timing.UpstreamStartAt.IsZero() || !timing.CompletedAt.IsZero() {
		t.Errorf("upstream stages set on a request that never reached it: %+v", timing)
	}
	if _, ok := timing.UpstreamDuration(); ok {
		t.Error("UpstreamDuration reports a value although the upstream was never called")
	}
}

func TestCarriesOutput(t *testing.T) {
	cases := []struct {
		name  string
		chunk llm.StreamChunk
		want  bool
	}{
		{"nil", nil, false},
		{"role only", llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "")), false},
		{"text", llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "hi")), true},
		{"tool call", llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "", llm.NewToolCallDelta(0, "call_1", "lookup", ""))), true},
		{"reasoning", llm.NewStreamChunk(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", "thinking", nil)), true},
		{"completion", llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(1, 1, 2)), false},
		{"error", llm.NewErrorStreamChunk(errors.New("boom")), false},
		{"relayed message_start", llm.NewRawEventChunk([]byte("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"), nil, false), false},
		{"relayed content delta", llm.NewRawEventChunk([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n"), nil, false), true},
		{"relayed content delta without event line", llm.NewRawEventChunk([]byte("data: {\"type\": \"content_block_delta\"}\n\n"), nil, false), true},
		{"relayed empty text block start", llm.NewRawEventChunk([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"), nil, false), false},
		{"relayed text block start with text", llm.NewRawEventChunk([]byte("data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\",\"text\":\"Hi\"}}\n\n"), nil, false), true},
		{"relayed tool use start", llm.NewRawEventChunk([]byte("data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"lookup\",\"input\":{}}}\n\n"), nil, false), true},
		{"relayed redacted thinking start", llm.NewRawEventChunk([]byte("data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"abc\"}}\n\n"), nil, false), true},
		{"relayed event naming a delta in its payload", llm.NewRawEventChunk([]byte("data: {\"type\":\"message_start\",\"message\":{\"model\":\"content_block_delta\",\"content\":[]}}\n\n"), nil, false), false},
		{"relayed content delta over several data lines", llm.NewRawEventChunk([]byte("event: content_block_delta\ndata: {\"type\":\ndata: \"content_block_delta\"}\n\n"), nil, false), true},
		{"relayed undecodable event", llm.NewRawEventChunk([]byte("data: content_block_delta\n\n"), nil, false), false},
		{"audio", llm.NewStreamChunk(llm.NewAudioStreamDelta(llm.RoleAssistant, "", "UklGRg==", "")), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := carriesOutput(c.chunk); got != c.want {
				t.Errorf("carriesOutput = %v, want %v", got, c.want)
			}
		})
	}
}

func TestExchangeTiming_Durations(t *testing.T) {
	start := time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC)
	timing := ExchangeTiming{
		ReceivedAt:      start,
		UpstreamStartAt: start.Add(10 * time.Millisecond),
		FirstOutputAt:   start.Add(260 * time.Millisecond),
		CompletedAt:     start.Add(1260 * time.Millisecond),
	}
	for name, got := range map[string]func() (time.Duration, bool){
		"routing":    timing.RoutingDuration,
		"ttfo":       timing.TimeToFirstOutput,
		"generation": timing.GenerationDuration,
		"upstream":   timing.UpstreamDuration,
		"total":      timing.TotalDuration,
	} {
		want := map[string]time.Duration{
			"routing": 10 * time.Millisecond, "ttfo": 250 * time.Millisecond,
			"generation": time.Second, "upstream": 1250 * time.Millisecond, "total": 1260 * time.Millisecond,
		}[name]
		if d, ok := got(); !ok || d != want {
			t.Errorf("%s = %s, %v; want %s, true", name, d, ok, want)
		}
	}

	if _, ok := (ExchangeTiming{ReceivedAt: start}).TotalDuration(); ok {
		t.Error("TotalDuration reports a value without CompletedAt")
	}
	if _, ok := (ExchangeTiming{UpstreamStartAt: start, CompletedAt: start.Add(-time.Second)}).UpstreamDuration(); ok {
		t.Error("UpstreamDuration reports a value for stages out of order")
	}
}

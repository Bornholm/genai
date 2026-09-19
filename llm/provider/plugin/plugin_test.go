package plugin

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/env"
	"github.com/bornholm/genai/llm/provider/plugin/protocol"
	"github.com/bornholm/genai/llm/retry"
)

// testPluginPath is the binary built by TestMain from plugin/sdk/internal/testplugin.
var testPluginPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "genai-plugin-test-*")
	if err != nil {
		panic(err)
	}
	testPluginPath = filepath.Join(dir, protocol.BinaryPrefix+"test")

	build := exec.Command("go", "build", "-o", testPluginPath, "./internal/testplugin")
	build.Dir = filepath.Join("..", "..", "..", "plugin", "sdk")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		panic("could not build test plugin: " + err.Error())
	}

	SetSearchDir(dir)
	code := m.Run()
	CleanupClients()
	os.RemoveAll(dir)
	os.Exit(code)
}

func newTestChatClient(t *testing.T, options map[string]string) *ChatCompletionClient {
	t.Helper()
	client, err := NewChatCompletionClient(context.Background(), testPluginPath, options)
	if err != nil {
		t.Fatalf("could not configure chat client: %+v", err)
	}
	return client
}

func TestResolve(t *testing.T) {
	path, err := Resolve("test")
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if path != testPluginPath {
		t.Errorf("expected %q, got %q", testPluginPath, path)
	}

	if _, err := Resolve("does-not-exist"); err == nil || !strings.Contains(err.Error(), "plugin not found") {
		t.Errorf("expected plugin not found error, got %v", err)
	}
	if _, err := Resolve("Bad Name"); err == nil {
		t.Error("expected an error for an invalid name")
	}
}

func TestChatCompletion(t *testing.T) {
	client := newTestChatClient(t, map[string]string{"MODEL": "m1"})

	tool := llm.NewFuncTool("get_weather", "weather", llm.NewJSONSchema().RequiredProperty("city", "city", "string"), nil)
	res, err := client.ChatCompletion(context.Background(),
		llm.WithMessages(llm.NewMessage(llm.RoleSystem, "sys"), llm.NewMessage(llm.RoleUser, "hello world")),
		llm.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}

	if got := res.Message().Content(); got != "echo(m1): hello world" {
		t.Errorf("unexpected content %q", got)
	}
	if res.Message().Role() != llm.RoleAssistant {
		t.Errorf("unexpected role %q", res.Message().Role())
	}

	rr, ok := res.(llm.ReasoningChatCompletionResponse)
	if !ok || rr.Reasoning() != "thinking" {
		t.Errorf("expected reasoning to survive the round trip, got %#v", res)
	}

	if len(res.ToolCalls()) != 1 {
		t.Fatalf("expected one tool call, got %d", len(res.ToolCalls()))
	}
	tc := res.ToolCalls()[0]
	if tc.ID() != "call-1" || tc.Name() != "get_weather" {
		t.Errorf("unexpected tool call %q %q", tc.ID(), tc.Name())
	}
	toolMessage, err := llm.ExecuteToolCall(context.Background(), tc, llm.NewFuncTool("get_weather", "weather", nil, func(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
		return llm.NewToolResult("city=" + params["city"].(string)), nil
	}))
	if err != nil {
		t.Fatalf("could not execute tool call: %+v", err)
	}
	if toolMessage.Content() != "city=Paris" {
		t.Errorf("unexpected tool result %q", toolMessage.Content())
	}

	usage := res.Usage()
	if usage.PromptTokens() != 3 || usage.CompletionTokens() != 5 || usage.TotalTokens() != 8 {
		t.Errorf("unexpected usage %d/%d/%d", usage.PromptTokens(), usage.CompletionTokens(), usage.TotalTokens())
	}
	if cc, ok := usage.(llm.CacheCreationReportingUsage); !ok || cc.CacheCreationTokens() != 2 {
		t.Errorf("expected cache creation tokens to survive, got %#v", usage)
	}
	if cr, ok := usage.(llm.CostReportingUsage); ok {
		if amount, currency, ok := cr.Cost(); !ok || amount != 0.5 || currency != "USD" {
			t.Errorf("unexpected cost %v %q %v", amount, currency, ok)
		}
	} else {
		t.Error("expected cost reporting usage")
	}
}

func TestChatCompletionWithAttachment(t *testing.T) {
	client := newTestChatClient(t, map[string]string{"MODEL": "m1"})

	image, err := llm.NewImageAttachment("image/png", "iVBORw0KGgo=", false)
	if err != nil {
		t.Fatal(err)
	}
	res, err := client.ChatCompletion(context.Background(),
		llm.WithMessages(llm.NewMultimodalMessage(llm.RoleUser, "look", image)),
	)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if !strings.HasSuffix(res.Message().Content(), "+attachment") {
		t.Errorf("attachment did not reach the plugin: %q", res.Message().Content())
	}
}

func TestChatCompletionStream(t *testing.T) {
	client := newTestChatClient(t, map[string]string{"MODEL": "m2"})

	tool := llm.NewFuncTool("get_weather", "weather", nil, nil)
	chunks, err := client.ChatCompletionStream(context.Background(),
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "hello world")),
		llm.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}

	var (
		content   strings.Builder
		reasoning strings.Builder
		params    strings.Builder
		complete  bool
	)
	tracker := llm.NewStreamingUsageTracker()
	for chunk := range chunks {
		tracker.Update(chunk)
		switch chunk.Type() {
		case llm.StreamChunkTypeError:
			t.Fatalf("unexpected error chunk: %+v", chunk.Error())
		case llm.StreamChunkTypeComplete:
			complete = true
		case llm.StreamChunkTypeDelta:
			delta := chunk.Delta()
			content.WriteString(delta.Content())
			if rd, ok := delta.(llm.ReasoningStreamDelta); ok {
				reasoning.WriteString(rd.Reasoning())
			}
			for _, tc := range delta.ToolCalls() {
				params.WriteString(tc.ParametersDelta())
			}
		}
	}

	if !complete {
		t.Error("stream ended without a complete chunk")
	}
	if content.String() != "echo(m2): hello world" {
		t.Errorf("unexpected content %q", content.String())
	}
	if reasoning.String() != "thinking" {
		t.Errorf("unexpected reasoning %q", reasoning.String())
	}
	if params.String() != `{"city":"Paris"}` {
		t.Errorf("unexpected tool call parameters %q", params.String())
	}
	if !tracker.Reported() || tracker.Usage().TotalTokens() != 8 {
		t.Errorf("usage did not survive the stream: %#v", tracker.Usage())
	}
}

func TestChatCompletionStreamCancellation(t *testing.T) {
	client := newTestChatClient(t, map[string]string{"HANG": "true"})

	ctx, cancel := context.WithCancel(context.Background())
	chunks, err := client.ChatCompletionStream(ctx, llm.WithMessages(llm.NewMessage(llm.RoleUser, "hello")))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}

	// First delta arrives, then the plugin blocks.
	select {
	case <-chunks:
	case <-time.After(5 * time.Second):
		t.Fatal("no first chunk")
	}

	cancel()

	var last llm.StreamChunk
	deadline := time.After(5 * time.Second)
	for {
		select {
		case chunk, ok := <-chunks:
			if !ok {
				if last == nil || last.Type() != llm.StreamChunkTypeError || !errors.Is(last.Error(), context.Canceled) {
					t.Errorf("expected a terminal chunk carrying context.Canceled, got %#v", last)
				}
				return
			}
			last = chunk
		case <-deadline:
			t.Fatal("stream did not close after cancellation")
		}
	}
}

func TestRateLimitErrorIsRetryable(t *testing.T) {
	client := newTestChatClient(t, map[string]string{"FAIL_WITH": "rate_limit"})

	_, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "hello")))
	if err == nil {
		t.Fatal("expected an error")
	}
	if !llm.IsRetryable(err) {
		t.Errorf("expected a retryable error, got %+v", err)
	}
	var httpErr *llm.HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != 429 || httpErr.Body != "slow down" {
		t.Errorf("expected the http error to survive, got %+v", err)
	}

	chunks, err := client.ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "hello")))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	var last llm.StreamChunk
	for chunk := range chunks {
		last = chunk
	}
	if last == nil || last.Type() != llm.StreamChunkTypeError || !llm.IsRetryable(last.Error()) {
		t.Errorf("expected a retryable error chunk, got %#v", last)
	}
}

func TestRetryDecoratorRetriesPluginErrors(t *testing.T) {
	client := newTestChatClient(t, map[string]string{"FAIL_WITH": "rate_limit"})
	wrapped := retry.NewClient(provider.NewClient(client, nil, nil), time.Millisecond, 3)

	start := time.Now()
	_, err := wrapped.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "hello")))
	if err == nil {
		t.Fatal("expected an error")
	}
	if time.Since(start) < 3*time.Millisecond {
		t.Error("retry decorator did not retry")
	}
}

func TestConfigureValidationError(t *testing.T) {
	_, err := NewChatCompletionClient(context.Background(), testPluginPath, map[string]string{"FAIL_WITH": "validation"})
	if err == nil {
		t.Fatal("expected an error")
	}
	var validationErr llm.ValidationError
	if !errors.As(err, &validationErr) || validationErr.Field != "model" {
		t.Errorf("expected the validation error to survive, got %+v", err)
	}
}

func TestEmbeddings(t *testing.T) {
	client, err := NewEmbeddingsClient(context.Background(), testPluginPath, nil)
	if err != nil {
		t.Fatalf("could not configure embeddings client: %+v", err)
	}
	res, err := client.Embeddings(context.Background(), []string{"ab", "abcd"}, llm.WithDimensions(2))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if len(res.Embeddings()) != 2 || len(res.Embeddings()[0]) != 2 || res.Embeddings()[1][0] != 4 {
		t.Errorf("unexpected embeddings %v", res.Embeddings())
	}
	if res.Usage().PromptTokens() != 2 {
		t.Errorf("unexpected usage %d", res.Usage().PromptTokens())
	}
}

func TestSharedProcess(t *testing.T) {
	chat := newTestChatClient(t, nil)
	embeddings, err := NewEmbeddingsClient(context.Background(), testPluginPath, nil)
	if err != nil {
		t.Fatalf("could not configure embeddings client: %+v", err)
	}
	if chat.Process() != embeddings.Process() {
		t.Error("expected chat and embeddings clients to share one process")
	}
	if chat.Process().Info().GetName() != "test" {
		t.Errorf("unexpected plugin name %q", chat.Process().Info().GetName())
	}
}

func TestRegistryFallbackFromEnv(t *testing.T) {
	t.Setenv("PLUGTEST_CHAT_COMPLETION_PROVIDER", "test")
	t.Setenv("PLUGTEST_CHAT_COMPLETION_TEST_MODEL", "from-env")
	t.Setenv("PLUGTEST_EMBEDDINGS_PROVIDER", "test")
	t.Setenv("PLUGTEST_EMBEDDINGS_TEST_COMMAND", testPluginPath)

	client, err := provider.Create(context.Background(), env.With("PLUGTEST_"))
	if err != nil {
		t.Fatalf("could not create client: %+v", err)
	}

	res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "hi")))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if res.Message().Content() != "echo(from-env): hi" {
		t.Errorf("options did not reach the plugin: %q", res.Message().Content())
	}

	if _, err := client.Embeddings(context.Background(), []string{"x"}); err != nil {
		t.Fatalf("unexpected embeddings error: %+v", err)
	}
}

func TestRegistryFallbackProgrammatic(t *testing.T) {
	client, err := provider.Create(context.Background(),
		provider.WithChatCompletion(provider.Name("test"), *NewOptions("test", map[string]string{"MODEL": "prog"})),
	)
	if err != nil {
		t.Fatalf("could not create client: %+v", err)
	}
	res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "hi")))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if res.Message().Content() != "echo(prog): hi" {
		t.Errorf("options did not reach the plugin: %q", res.Message().Content())
	}
}

func TestFallbackInactiveWithoutSearchDir(t *testing.T) {
	dir := SearchDir()
	SetSearchDir("")
	t.Cleanup(func() { SetSearchDir(dir) })

	_, err := provider.Create(context.Background(),
		provider.WithChatCompletion(provider.Name("test"), *NewOptions("test", nil)),
	)
	if !errors.Is(err, provider.ErrClientNotFound) {
		t.Errorf("expected ErrClientNotFound without a plugin directory, got %v", err)
	}
}

func TestClientSurvivesPluginDeath(t *testing.T) {
	client := newTestChatClient(t, map[string]string{"MODEL": "phoenix"})
	first := client.Process()

	first.Kill()
	// Killing is asynchronous on the go-plugin side; wait for it to notice.
	deadline := time.Now().Add(5 * time.Second)
	for !first.Exited() && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if !first.Exited() {
		t.Fatal("process did not exit")
	}

	res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "back")))
	if err != nil {
		t.Fatalf("expected the client to reconfigure on a fresh process, got %+v", err)
	}
	if res.Message().Content() != "echo(phoenix): back" {
		t.Errorf("options were not reapplied: %q", res.Message().Content())
	}
	if client.Process() == first {
		t.Error("expected a new process")
	}
}

func TestCloseReleasesClient(t *testing.T) {
	client := newTestChatClient(t, nil)
	if err := client.Close(); err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	_, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "hi")))
	if !errors.Is(err, llm.ErrUnavailable) {
		t.Errorf("expected ErrUnavailable after Close, got %v", err)
	}
	if err := client.Close(); err != nil {
		t.Errorf("second Close should be a no-op, got %v", err)
	}
}

func TestUnknownPluginError(t *testing.T) {
	_, err := provider.Create(context.Background(),
		provider.WithChatCompletion(provider.Name("nope"), *NewOptions("nope", nil)),
	)
	if err == nil || !strings.Contains(err.Error(), "plugin not found") {
		t.Errorf("expected plugin not found, got %v", err)
	}
}

package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

func TestChatCompletion_ReadsThoughtsToolCallsSignaturesAndUsage(t *testing.T) {
	var path, key string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path, key = r.URL.Path, r.Header.Get("x-goog-api-key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[
			{"text":"Considering the cities.","thought":true,"thoughtSignature":"sig-thought"},
			{"text":"Checking both."},
			{"functionCall":{"name":"weather","args":{"city":"Paris"}},"thoughtSignature":"sig-1"},
			{"functionCall":{"id":"fc-2","name":"weather","args":{"city":"Lyon"}}}
		]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":60,"candidatesTokenCount":20,"thoughtsTokenCount":30,"totalTokenCount":150}}`)
	}))
	defer server.Close()

	res, err := NewChatCompletionClient(nil, server.URL+"/v1beta/", "key", "gemini-test").
		ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Weather?")))
	if err != nil {
		t.Fatalf("chat completion: %v", err)
	}

	if path != "/v1beta/models/gemini-test:generateContent" || key != "key" {
		t.Errorf("path %q, key %q", path, key)
	}
	if res.Message().Content() != "Checking both." {
		t.Errorf("content = %q", res.Message().Content())
	}

	calls := res.ToolCalls()
	if len(calls) != 2 || !strings.HasPrefix(calls[0].ID(), "call_") || calls[0].Parameters() != `{"city":"Paris"}` || calls[1].ID() != "fc-2" {
		t.Fatalf("tool calls = %v", calls)
	}

	reasoning := res.(llm.ReasoningChatCompletionResponse)
	if reasoning.Reasoning() != "Considering the cities." {
		t.Errorf("reasoning = %q", reasoning.Reasoning())
	}
	details := reasoning.ReasoningDetails()
	if len(details) != 1 || details[0].ID != calls[0].ID() || details[0].Data != "sig-1" || details[0].Format != signatureFormat {
		t.Errorf("details = %+v", details)
	}

	usage := res.Usage()
	if usage.PromptTokens() != 100 || usage.CompletionTokens() != 50 || usage.TotalTokens() != 150 {
		t.Errorf("usage = %d/%d/%d", usage.PromptTokens(), usage.CompletionTokens(), usage.TotalTokens())
	}
}

func TestChatCompletion_ReportsBlocksAndErrors(t *testing.T) {
	cases := map[string]struct {
		status    int
		body      string
		want      int
		noMessage bool
	}{
		"rejected request": {status: http.StatusBadRequest, body: `{"error":{"code":400,"message":"bad","status":"INVALID_ARGUMENT"}}`, want: http.StatusBadRequest},
		"overloaded":       {status: http.StatusServiceUnavailable, body: `{"error":{"code":503,"message":"overloaded"}}`, want: http.StatusServiceUnavailable},
		"blocked prompt":   {status: http.StatusOK, body: `{"promptFeedback":{"blockReason":"SAFETY"}}`, want: http.StatusBadRequest},
		"blocked answer":   {status: http.StatusOK, body: `{"candidates":[{"finishReason":"SAFETY"}]}`, want: http.StatusBadRequest},
		"malformed call":   {status: http.StatusOK, body: `{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL"}]}`, want: http.StatusBadGateway},
		"thoughts only":    {status: http.StatusOK, body: `{"candidates":[{"content":{"parts":[{"text":"hmm","thought":true}]},"finishReason":"MAX_TOKENS"}]}`, noMessage: true},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(c.status)
				_, _ = io.WriteString(w, c.body)
			}))
			defer server.Close()

			_, err := NewChatCompletionClient(nil, server.URL, "", "gemini-test").
				ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))

			if c.noMessage {
				if !errors.Is(err, llm.ErrNoMessage) {
					t.Fatalf("error = %v, want ErrNoMessage", err)
				}
				return
			}
			var httpErr *llm.HTTPError
			if !errors.As(err, &httpErr) || httpErr.StatusCode != c.want {
				t.Fatalf("error = %v, want status %d", err, c.want)
			}
		})
	}
}

// streamServer serves chunks as an SSE stream, the way streamGenerateContent
// does with alt=sse: data lines only, no terminal event.
func streamServer(t *testing.T, chunks ...string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":streamGenerateContent") || r.URL.Query().Get("alt") != "sse" {
			http.Error(w, "stream expected", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range chunks {
			_, _ = io.WriteString(w, "data: "+chunk+"\r\n\r\n")
		}
	}))
}

type collected struct {
	content   string
	reasoning string
	details   []llm.ReasoningDetail
	calls     []llm.ToolCallDelta
	terminal  llm.StreamChunk
}

func collect(t *testing.T, server *httptest.Server) collected {
	t.Helper()
	chunks, err := NewChatCompletionClient(nil, server.URL, "", "gemini-test").
		ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var result collected
	for chunk := range chunks {
		if chunk.Type() != llm.StreamChunkTypeDelta {
			result.terminal = chunk
			continue
		}
		delta := chunk.Delta()
		result.content += delta.Content()
		if rd, ok := delta.(interface {
			Reasoning() string
			ReasoningDetails() []llm.ReasoningDetail
		}); ok {
			result.reasoning += rd.Reasoning()
			result.details = append(result.details, rd.ReasoningDetails()...)
		}
		result.calls = append(result.calls, delta.ToolCalls()...)
	}
	if result.terminal == nil {
		t.Fatal("the stream ended without a terminal chunk")
	}
	return result
}

func TestChatCompletionStream_StreamsThoughtsTextAndToolCalls(t *testing.T) {
	server := streamServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Thinking.","thought":true}]}}],"usageMetadata":{"promptTokenCount":10}}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello, "}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"world.","thoughtSignature":"sig-text"}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fc-1","name":"ping","args":{}},"thoughtSignature":"sig-call"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":3,"totalTokenCount":18}}`,
	)
	defer server.Close()

	got := collect(t, server)

	if got.content != "Hello, world." || got.reasoning != "Thinking." {
		t.Errorf("content %q, reasoning %q", got.content, got.reasoning)
	}
	if len(got.calls) != 1 || got.calls[0].ID() != "fc-1" || got.calls[0].Name() != "ping" || got.calls[0].ParametersDelta() != "{}" {
		t.Errorf("calls = %v", got.calls)
	}
	if len(got.details) != 2 || got.details[0].ID != "" || got.details[1].ID != "fc-1" || got.details[1].Index != 1 {
		t.Errorf("details = %+v", got.details)
	}
	if !got.terminal.IsComplete() || got.terminal.Usage().CompletionTokens() != 8 || got.terminal.Usage().TotalTokens() != 18 {
		t.Errorf("terminal = %v", got.terminal)
	}
}

func TestChatCompletionStream_EndsOnFailures(t *testing.T) {
	cases := map[string]struct {
		chunks    []string
		status    int
		noMessage bool
		complete  bool
	}{
		"cut stream": {
			chunks: []string{`{"candidates":[{"content":{"parts":[{"text":"Hel"}]}}],"usageMetadata":{"promptTokenCount":10}}`},
			status: http.StatusBadGateway,
		},
		"error in the body": {
			chunks: []string{`{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}`},
			status: http.StatusTooManyRequests,
		},
		"blocked answer": {
			chunks: []string{`{"candidates":[{"finishReason":"SAFETY"}]}`},
			status: http.StatusBadRequest,
		},
		"safety stop after some text": {
			chunks:   []string{`{"candidates":[{"content":{"parts":[{"text":"Partial"}]},"finishReason":"SAFETY"}]}`},
			complete: true,
		},
		"nothing produced": {
			chunks:    []string{`{"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}]}`},
			noMessage: true,
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			server := streamServer(t, c.chunks...)
			defer server.Close()

			got := collect(t, server)
			switch {
			case c.complete:
				if !got.terminal.IsComplete() {
					t.Fatalf("terminal = %v, want complete", got.terminal)
				}
			case c.noMessage:
				if !errors.Is(got.terminal.Error(), llm.ErrNoMessage) {
					t.Fatalf("error = %v, want ErrNoMessage", got.terminal.Error())
				}
			default:
				var httpErr *llm.HTTPError
				if !errors.As(got.terminal.Error(), &httpErr) || httpErr.StatusCode != c.status {
					t.Fatalf("error = %v, want status %d", got.terminal.Error(), c.status)
				}
			}
		})
	}
}

func TestRegisteredFactories(t *testing.T) {
	var chatPath, listQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Header.Get("x-goog-api-key") != "key" {
			http.Error(w, "no key", http.StatusUnauthorized)
			return
		}
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1beta/models":
			listQuery = r.URL.RawQuery
			if r.URL.Query().Get("pageToken") == "" {
				_, _ = io.WriteString(w, `{"models":[{"name":"models/gemini-3-pro","displayName":"Gemini 3 Pro","inputTokenLimit":1048576,"outputTokenLimit":65536}],"nextPageToken":"p2"}`)
				return
			}
			_, _ = io.WriteString(w, `{"models":[{"name":"models/gemini-3-flash"}]}`)
		default:
			chatPath = r.URL.Path
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			_, _ = io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"pong"}]},"finishReason":"STOP"}]}`)
		}
	}))
	defer server.Close()

	common := provider.CommonOptions{BaseURL: server.URL + "/v1beta", APIKey: "key", Model: "models/gemini-3-pro"}
	client, err := provider.Create(context.Background(),
		provider.WithChatCompletion(Name, Options{CommonOptions: common}),
		provider.WithDiscovery(Name, Options{CommonOptions: common}),
	)
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "ping")))
	if err != nil || res.Message().Content() != "pong" || chatPath != "/v1beta/models/gemini-3-pro:generateContent" {
		t.Fatalf("chat: %v, path %q", err, chatPath)
	}

	models, err := client.(llm.DiscoveryClient).Models(context.Background())
	if err != nil {
		t.Fatalf("models: %v", err)
	}
	if len(models) != 2 || models[0].ID() != "gemini-3-pro" || models[0].ContextWindow() != 1048576 ||
		models[0].MaxOutputTokens() != 65536 || models[1].ID() != "gemini-3-flash" || !strings.Contains(listQuery, "pageToken=p2") {
		t.Errorf("models = %v, last query %q", models, listQuery)
	}
}

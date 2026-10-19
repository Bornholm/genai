package responses

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

func TestChatCompletion_ReadsTextReasoningToolCallsAndUsage(t *testing.T) {
	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" || r.Header.Get("Authorization") != "Bearer sk-test" {
			http.Error(w, "unexpected request", http.StatusBadRequest)
			return
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_1","status":"completed","output":[
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"First."},{"type":"summary_text","text":"Second."}],"encrypted_content":"gAAA"},
			{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"Let me check."}]},
			{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Paris\"}"}
		],"usage":{"input_tokens":120,"input_tokens_details":{"cached_tokens":100},"output_tokens":30,"total_tokens":150}}`)
	}))
	defer server.Close()

	client := NewChatCompletionClient(nil, server.URL+"/v1/", "sk-test", "gpt-test")
	res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Weather?")))
	if err != nil {
		t.Fatalf("chat completion: %v", err)
	}

	if got["stream"] != nil {
		t.Errorf("a plain completion must not ask for a stream, got %v", got["stream"])
	}
	if res.Message().Content() != "Let me check." {
		t.Errorf("content = %q", res.Message().Content())
	}
	if calls := res.ToolCalls(); len(calls) != 1 || calls[0].ID() != "call_1" || calls[0].Name() != "weather" || calls[0].Parameters() != `{"city":"Paris"}` {
		t.Errorf("tool calls = %v", calls)
	}

	reasoning := res.(llm.ReasoningChatCompletionResponse)
	if reasoning.Reasoning() != "First.\n\nSecond." {
		t.Errorf("reasoning = %q", reasoning.Reasoning())
	}
	details := reasoning.ReasoningDetails()
	if len(details) != 3 || details[2].Type != llm.ReasoningDetailTypeEncrypted || details[2].Data != "gAAA" || details[2].ID != "rs_1" || details[2].Index != 2 {
		t.Errorf("details = %+v", details)
	}

	usage := res.Usage()
	if usage.PromptTokens() != 120 || usage.CompletionTokens() != 30 || usage.TotalTokens() != 150 {
		t.Errorf("usage = %d/%d/%d", usage.PromptTokens(), usage.CompletionTokens(), usage.TotalTokens())
	}
	if cached := usage.(interface{ CachedTokens() int64 }).CachedTokens(); cached != 100 {
		t.Errorf("cached tokens = %d", cached)
	}
}

func TestChatCompletion_ReportsUpstreamErrors(t *testing.T) {
	cases := map[string]struct {
		handler   http.HandlerFunc
		status    int
		retryable bool
	}{
		"rejected request": {
			handler: func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, `{"error":{"message":"bad"}}`, http.StatusBadRequest)
			},
			status: http.StatusBadRequest,
		},
		"rate limit": {
			handler: func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, `{"error":{"message":"slow down"}}`, http.StatusTooManyRequests)
			},
			status:    http.StatusTooManyRequests,
			retryable: true,
		},
		"failed response": {
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"status":"failed","error":{"code":"server_error","message":"boom"},"output":[]}`)
			},
			status:    http.StatusBadGateway,
			retryable: true,
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(c.handler)
			defer server.Close()

			_, err := NewChatCompletionClient(nil, server.URL, "", "gpt-test").
				ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))

			var httpErr *llm.HTTPError
			if !errors.As(err, &httpErr) || httpErr.StatusCode != c.status {
				t.Fatalf("error = %v, want status %d", err, c.status)
			}
			if llm.IsRetryable(err) != c.retryable {
				t.Errorf("retryable = %v, want %v", llm.IsRetryable(err), c.retryable)
			}
		})
	}
}

func TestChatCompletion_EmptyOutputIsNoMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"status":"completed","output":[{"type":"reasoning","id":"rs_1","summary":[]}]}`)
	}))
	defer server.Close()

	_, err := NewChatCompletionClient(nil, server.URL, "", "gpt-test").
		ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))
	if !errors.Is(err, llm.ErrNoMessage) {
		t.Fatalf("error = %v, want ErrNoMessage", err)
	}
}

// streamServer serves events as a response stream.
func streamServer(t *testing.T, events ...string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["stream"] != true {
			http.Error(w, "stream expected", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			var parsed struct {
				Type string `json:"type"`
			}
			_ = json.Unmarshal([]byte(event), &parsed)
			_, _ = io.WriteString(w, "event: "+parsed.Type+"\ndata: "+event+"\n\n")
		}
	}))
}

type collected struct {
	content   string
	reasoning string
	details   []llm.ReasoningDetail
	toolCalls map[int]*struct{ id, name, args string }
	terminal  llm.StreamChunk
}

func collect(t *testing.T, server *httptest.Server, funcs ...llm.ChatCompletionOptionFunc) collected {
	t.Helper()
	funcs = append([]llm.ChatCompletionOptionFunc{llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi"))}, funcs...)
	chunks, err := NewChatCompletionClient(nil, server.URL, "", "gpt-test").ChatCompletionStream(context.Background(), funcs...)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	result := collected{toolCalls: map[int]*struct{ id, name, args string }{}}
	for chunk := range chunks {
		if chunk.Type() != llm.StreamChunkTypeDelta {
			if result.terminal != nil {
				t.Fatalf("a second terminal chunk: %v", chunk)
			}
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
		for _, tc := range delta.ToolCalls() {
			call, ok := result.toolCalls[tc.Index()]
			if !ok {
				call = &struct{ id, name, args string }{}
				result.toolCalls[tc.Index()] = call
			}
			if tc.ID() != "" {
				call.id = tc.ID()
			}
			if tc.Name() != "" {
				call.name = tc.Name()
			}
			call.args += tc.ParametersDelta()
		}
	}
	if result.terminal == nil {
		t.Fatal("the stream ended without a terminal chunk")
	}
	return result
}

func TestChatCompletionStream_StreamsTextReasoningAndToolCalls(t *testing.T) {
	server := streamServer(t,
		`{"type":"response.created","response":{"id":"resp_1","status":"in_progress","output":[],"usage":null}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`,
		`{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"First"}`,
		`{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"."}`,
		`{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":1,"delta":"Second."}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"First."},{"type":"summary_text","text":"Second."}],"encrypted_content":"gAAA"}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}`,
		`{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"Let me "}`,
		`{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"check."}`,
		`{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"Let me check."}]}}`,
		`{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","output_index":2,"delta":"{\"city\":"}`,
		`{"type":"response.function_call_arguments.delta","output_index":2,"delta":"\"Paris\"}"}`,
		`{"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Paris\"}"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":120,"input_tokens_details":{"cached_tokens":100},"output_tokens":30,"total_tokens":150}}}`,
	)
	defer server.Close()

	got := collect(t, server)

	if got.content != "Let me check." {
		t.Errorf("content = %q", got.content)
	}
	if got.reasoning != "First.\n\nSecond." {
		t.Errorf("reasoning = %q", got.reasoning)
	}
	if len(got.details) != 3 || got.details[2].Data != "gAAA" || got.details[2].Format != reasoningFormat {
		t.Errorf("details = %+v", got.details)
	}
	call := got.toolCalls[0]
	if len(got.toolCalls) != 1 || call.id != "call_1" || call.name != "weather" || call.args != `{"city":"Paris"}` {
		t.Errorf("tool calls = %+v", got.toolCalls)
	}
	if !got.terminal.IsComplete() || got.terminal.Usage().PromptTokens() != 120 || got.terminal.Usage().TotalTokens() != 150 {
		t.Errorf("terminal = %v", got.terminal)
	}
}

func TestChatCompletionStream_TakesWholeItemsFromAGateway(t *testing.T) {
	server := streamServer(t,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"message","content":[{"type":"output_text","text":"Hello."}]}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"ping","arguments":"{}"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{}"}`,
		`{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"ping","arguments":"{}"}}`,
		`{"type":"response.completed","response":{"status":"completed"}}`,
	)
	defer server.Close()

	got := collect(t, server)

	if got.content != "Hello." {
		t.Errorf("content = %q", got.content)
	}
	if call := got.toolCalls[0]; call == nil || call.args != "{}" {
		t.Errorf("arguments shipped whole must not be doubled, got %+v", call)
	}
	if !got.terminal.IsComplete() {
		t.Errorf("terminal = %v", got.terminal)
	}
}

func TestChatCompletionStream_HidesExcludedReasoningButKeepsItsDetails(t *testing.T) {
	server := streamServer(t,
		`{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"Secret."}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Secret."}],"encrypted_content":"gAAA"}}`,
		`{"type":"response.output_text.delta","output_index":1,"delta":"Done."}`,
		`{"type":"response.output_item.done","output_index":1,"item":{"type":"message","content":[{"type":"output_text","text":"Done."}]}}`,
		`{"type":"response.completed","response":{"status":"completed"}}`,
	)
	defer server.Close()

	got := collect(t, server, llm.WithReasoning(&llm.ReasoningOptions{Effort: ptr(llm.ReasoningEffortLow), Exclude: true}))

	if got.reasoning != "" {
		t.Errorf("reasoning = %q, want none", got.reasoning)
	}
	if len(got.details) != 2 {
		t.Errorf("details = %+v, want the summary and the encrypted content", got.details)
	}
}

func TestChatCompletionStream_EndsOnFailures(t *testing.T) {
	cases := map[string]struct {
		events    []string
		status    int
		usage     int64
		noMessage bool
	}{
		"failed response": {
			events: []string{
				`{"type":"response.output_text.delta","output_index":0,"delta":"Hel"}`,
				`{"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"boom"},"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}}}`,
			},
			status: http.StatusBadGateway,
			usage:  10,
		},
		"error event": {
			events: []string{
				`{"type":"error","code":"rate_limit_exceeded","message":"slow down"}`,
			},
			status: http.StatusTooManyRequests,
		},
		"truncated stream": {
			events: []string{
				`{"type":"response.output_text.delta","output_index":0,"delta":"Hel"}`,
			},
			status: http.StatusBadGateway,
		},
		"nothing produced": {
			events: []string{
				`{"type":"response.completed","response":{"status":"completed","output":[]}}`,
			},
			noMessage: true,
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			server := streamServer(t, c.events...)
			defer server.Close()

			got := collect(t, server)
			err := got.terminal.Error()
			if c.noMessage {
				if !errors.Is(err, llm.ErrNoMessage) {
					t.Fatalf("error = %v, want ErrNoMessage", err)
				}
				return
			}
			var httpErr *llm.HTTPError
			if !errors.As(err, &httpErr) || httpErr.StatusCode != c.status {
				t.Fatalf("error = %v, want status %d", err, c.status)
			}
			if c.usage > 0 && (got.terminal.Usage() == nil || got.terminal.Usage().PromptTokens() != c.usage) {
				t.Errorf("usage = %v, want the published one", got.terminal.Usage())
			}
		})
	}
}

func TestChatCompletionStream_ReturnsTheRejectionBeforeStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		http.Error(w, `{"error":{"message":"slow down"}}`, http.StatusTooManyRequests)
	}))
	defer server.Close()

	_, err := NewChatCompletionClient(nil, server.URL, "", "gpt-test").
		ChatCompletionStream(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "Hi")))

	var httpErr *llm.HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusTooManyRequests || httpErr.Header.Get("Retry-After") != "7" {
		t.Fatalf("error = %v, want the 429 with its headers", err)
	}
}

func TestRegisteredFactory(t *testing.T) {
	var path, auth, model string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path, auth = r.URL.Path, r.Header.Get("Authorization")
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		model, _ = body["model"].(string)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"pong"}]}]}`)
	}))
	defer server.Close()

	client, err := provider.Create(context.Background(), provider.WithChatCompletion(Name, Options{
		CommonOptions: provider.CommonOptions{BaseURL: server.URL + "/v1", APIKey: "sk-go", Model: "gpt-luna"},
	}))
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	res, err := client.ChatCompletion(context.Background(), llm.WithMessages(llm.NewMessage(llm.RoleUser, "ping")))
	if err != nil {
		t.Fatalf("chat completion: %v", err)
	}
	if res.Message().Content() != "pong" || path != "/v1/responses" || auth != "Bearer sk-go" || model != "gpt-luna" {
		t.Errorf("content %q, path %q, auth %q, model %q", res.Message().Content(), path, auth, model)
	}
	if !strings.HasPrefix(string(Name), "openai") {
		t.Errorf("name = %q", Name)
	}
}

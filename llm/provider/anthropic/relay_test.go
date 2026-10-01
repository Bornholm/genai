package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/circuitbreaker"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/ratelimit"
)

// relayFixture is a tool call turn shaped like Claude Code's auto mode
// traffic: the safety verdict rides on message_delta, keyed by tool_use ID.
const relayFixture = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-real","content":[],"stop_reason":null,"usage":{"input_tokens":2,"cache_creation_input_tokens":100,"cache_read_input_tokens":50,"output_tokens":1}}}` + "\n\n" +
	"event: ping\n" + `data: {"type":"ping"}` + "\n\n" +
	"event: content_block_start\n" +
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"Bash","input":{}}}` + "\n\n" +
	"event: content_block_delta\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"ls <dir>\"}"}}` + "\n\n" +
	"event: content_block_stop\n" + `data: {"type":"content_block_stop","index":0}` + "\n\n" +
	"event: message_delta\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null,"safeguard_results":[{"type":"dangerous_tool_use","status":{"type":"available","tool_uses":{"toolu_01":{"type":"evaluated","outcome":"not_flagged"}}}}]},"usage":{"output_tokens":94}}` + "\n\n" +
	"event: message_stop\n" + `data: {"type":"message_stop"}` + "\n\n"

const relayRequest = `{"model":"org/alias","max_tokens":64,"stream":true,` +
	`"messages":[{"role":"user","content":"run <this>"}],` +
	`"safeguards":[{"type":"dangerous_tool_use","classifier_context":{"v":1,"permission_mode":"auto"}}]}`

func newRelayClient(baseURL string) *ChatCompletionClient {
	c := NewChatCompletionClient(anthropicsdk.NewClient(), "claude-real", 0)
	c.relay = &relayTarget{baseURL: normalizeBaseURL(baseURL), apiKey: "provider-key"}
	return c
}

func collect(t *testing.T, chunks <-chan llm.StreamChunk) []llm.StreamChunk {
	t.Helper()
	var all []llm.StreamChunk
	for chunk := range chunks {
		all = append(all, chunk)
	}
	return all
}

func TestRelayMessages_ForwardsRequestAndEventsVerbatim(t *testing.T) {
	var got *http.Request
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, relayFixture)
	}))
	defer srv.Close()

	header := http.Header{}
	header.Set("Anthropic-Beta", "dangerous-tool-use-2026-09-03,effort-2025-11-24")
	header.Set("Authorization", "Bearer gateway-token")
	header.Set("X-Claude-Code-Session-Id", "s1")

	chunks, err := newRelayClient(srv.URL+"/v1").RelayMessages(context.Background(), []byte(relayRequest), header)
	if err != nil {
		t.Fatal(err)
	}
	all := collect(t, chunks)

	if got.URL.Path != "/v1/messages" {
		t.Errorf("path = %q", got.URL.Path)
	}
	if got.Header.Get("x-api-key") != "provider-key" {
		t.Errorf("x-api-key = %q, want the provider key", got.Header.Get("x-api-key"))
	}
	if got.Header.Get("Authorization") != "" || got.Header.Get("X-Claude-Code-Session-Id") != "" {
		t.Errorf("non anthropic-* headers leaked upstream: %v", got.Header)
	}
	if got.Header.Get("anthropic-beta") != header.Get("Anthropic-Beta") {
		t.Errorf("anthropic-beta = %q", got.Header.Get("anthropic-beta"))
	}
	if got.Header.Get("anthropic-version") != "2023-06-01" {
		t.Errorf("anthropic-version = %q", got.Header.Get("anthropic-version"))
	}

	var sent, client map[string]any
	if err := json.Unmarshal(gotBody, &sent); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(relayRequest), &client); err != nil {
		t.Fatal(err)
	}
	if sent["model"] != "claude-real" {
		t.Errorf("model = %v, want the client's model", sent["model"])
	}
	// Everything but the model reaches the upstream as the client sent it.
	delete(sent, "model")
	delete(client, "model")
	if !reflect.DeepEqual(sent, client) {
		t.Errorf("body altered beyond the model:\nsent   %v\nclient %v", sent, client)
	}
	if !strings.Contains(string(gotBody), "run <this>") {
		t.Errorf("body was HTML-escaped: %s", gotBody)
	}

	var relayed strings.Builder
	for _, chunk := range all {
		relayed.Write(chunk.(llm.RawEventChunk).RawEvent())
	}
	if relayed.String() != relayFixture {
		t.Errorf("events not relayed verbatim:\n%s", relayed.String())
	}

	last := all[len(all)-1]
	if !last.IsComplete() || last.Error() != nil {
		t.Fatalf("last chunk complete=%v err=%v", last.IsComplete(), last.Error())
	}
	usage := last.Usage()
	if usage.PromptTokens() != 152 || usage.CompletionTokens() != 94 {
		t.Errorf("usage = %d prompt / %d completion, want 152 / 94", usage.PromptTokens(), usage.CompletionTokens())
	}
}

func TestRelayMessages_RejectionKeepsStatusHeadersAndBody(t *testing.T) {
	const body = `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()

	_, err := newRelayClient(srv.URL).RelayMessages(context.Background(), []byte(relayRequest), nil)

	var httpErr *llm.HTTPError
	if !errors.As(err, &httpErr) {
		t.Fatalf("err = %v, want an *llm.HTTPError", err)
	}
	if httpErr.StatusCode != http.StatusTooManyRequests || httpErr.Body != body || httpErr.Header.Get("Retry-After") != "7" {
		t.Errorf("got %d %q retry-after=%q", httpErr.StatusCode, httpErr.Body, httpErr.Header.Get("Retry-After"))
	}
	if !llm.IsRetryable(err) {
		t.Error("a 429 must stay retryable")
	}
}

func TestRelayMessages_ErrorEventIsRelayedAsIs(t *testing.T) {
	const errEvent = "event: error\n" + `data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}` + "\n\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: ping\n"+`data: {"type":"ping"}`+"\n\n"+errEvent)
	}))
	defer srv.Close()

	chunks, err := newRelayClient(srv.URL).RelayMessages(context.Background(), []byte(relayRequest), nil)
	if err != nil {
		t.Fatal(err)
	}
	all := collect(t, chunks)

	var rawErr *llm.RawEventError
	if !errors.As(all[len(all)-1].Error(), &rawErr) || string(rawErr.Event) != errEvent {
		t.Fatalf("last chunk error = %v, want the error event verbatim", all[len(all)-1].Error())
	}
	if !llm.IsRetryable(rawErr) {
		t.Error("an overloaded upstream must stay retryable")
	}
}

func TestRelayMessages_TruncatedStreamEndsWithoutTerminalChunk(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, relayFixture[:strings.Index(relayFixture, "event: content_block_stop")])
	}))
	defer srv.Close()

	chunks, err := newRelayClient(srv.URL).RelayMessages(context.Background(), []byte(relayRequest), nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, chunk := range collect(t, chunks) {
		if chunk.IsComplete() || chunk.Error() != nil {
			t.Fatalf("truncated stream produced a terminal chunk: complete=%v err=%v", chunk.IsComplete(), chunk.Error())
		}
	}
}

func TestSupportsMessagesRelay_OnlyForRegistryBuiltClients(t *testing.T) {
	if llm.SupportsMessagesRelay(NewChatCompletionClient(anthropicsdk.NewClient(), "m", 0)) {
		t.Error("a client built without its upstream cannot relay")
	}
	if !llm.SupportsMessagesRelay(newRelayClient("https://api.anthropic.com")) {
		t.Error("a registry-built client must relay")
	}
}

// TestRegistryBuiltClientRelaysThroughDecorators covers the chain a gateway
// actually holds: the registry's composite client under the stock rate
// limiter, relaying to the base URL and key the registry was given.
func TestRegistryBuiltClientRelaysThroughDecorators(t *testing.T) {
	var gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-api-key")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, relayFixture)
	}))
	defer srv.Close()

	opts := defaultOptions()
	opts.BaseURL = srv.URL + "/v1/"
	opts.APIKey = "registry-key"
	opts.Model = "claude-real"
	base, err := provider.Create(context.Background(), provider.WithChatCompletion(Name, *opts))
	if err != nil {
		t.Fatal(err)
	}
	client := circuitbreaker.NewClient(ratelimit.NewClient(base), 3, time.Minute)

	if !llm.SupportsMessagesRelay(client) {
		t.Fatal("the registry client must relay through the circuit breaker and the rate limiter")
	}
	chunks, err := llm.RelayMessages(context.Background(), client, []byte(relayRequest), nil)
	if err != nil {
		t.Fatal(err)
	}
	collect(t, chunks)
	if gotKey != "registry-key" {
		t.Errorf("x-api-key = %q", gotKey)
	}
}

func TestRelayMessages_LowercaseHeadersAreNotDuplicated(t *testing.T) {
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, relayFixture)
	}))
	defer srv.Close()

	header := http.Header{"anthropic-version": {"2024-01-01"}}
	chunks, err := newRelayClient(srv.URL).RelayMessages(context.Background(), []byte(relayRequest), header)
	if err != nil {
		t.Fatal(err)
	}
	collect(t, chunks)

	if v := got.Values("Anthropic-Version"); len(v) != 1 || v[0] != "2024-01-01" {
		t.Errorf("anthropic-version = %v, want the caller's value only", v)
	}
}

func TestWithModel_KeepsTheBytesOfABodyAlreadyNamingTheModel(t *testing.T) {
	const body = `{"stream":true,  "model":"claude-real","messages":[]}`
	got, err := withModel([]byte(body), "claude-real")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != body {
		t.Errorf("body rewritten: %s", got)
	}
}

// A redirect would carry x-api-key to whatever host it names: it is not
// followed, and comes back as the upstream's answer.
func TestRelayMessages_DoesNotFollowRedirects(t *testing.T) {
	var leaked bool
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		leaked = r.Header.Get("x-api-key") != ""
	}))
	defer elsewhere.Close()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, elsewhere.URL+"/v1/messages", http.StatusTemporaryRedirect)
	}))
	defer srv.Close()

	_, err := newRelayClient(srv.URL).RelayMessages(context.Background(), []byte(relayRequest), nil)

	var httpErr *llm.HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusTemporaryRedirect {
		t.Errorf("err = %v, want the 307 as an *llm.HTTPError", err)
	}
	if leaked {
		t.Error("the API key followed the redirect to another host")
	}
}

func TestWithModel_RefusesANullBody(t *testing.T) {
	if _, err := withModel([]byte("null"), "claude-real"); err == nil {
		t.Error("a null body must be refused, not panic")
	}
}

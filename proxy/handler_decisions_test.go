package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/circuitbreaker"
	"github.com/bornholm/genai/llm/ratelimit"
	"github.com/bornholm/genai/llm/retry"
)

// mockDecisionClient answers decisions and records what it was asked.
type mockDecisionClient struct {
	mockChatClient
	response  llm.DecisionResponse
	err       error
	state     any
	questions llm.Questions
}

func (m *mockDecisionClient) Decision(_ context.Context, state any, questions llm.Questions, _ ...llm.DecisionOptionFunc) (llm.DecisionResponse, error) {
	m.state, m.questions = state, questions
	return m.response, m.err
}

// decisionRecorder keeps the response the post-response hooks saw.
type decisionRecorder struct {
	req *ProxyRequest
	res *ProxyResponse
}

func (u *decisionRecorder) Name() string  { return "test.usage" }
func (u *decisionRecorder) Priority() int { return 0 }
func (u *decisionRecorder) PostResponse(_ context.Context, req *ProxyRequest, res *ProxyResponse) (*HookResult, error) {
	u.req, u.res = req, res
	return nil, nil
}

const decisionBody = `{
	"model": "jev",
	"state": {"ticket": "The checkout page returns a 500 since this morning."},
	"questions": {
		"is_urgent": {"type": "noul", "instructions": "Is this ticket urgent?", "criteria": {"true": "blocks customers"}},
		"team": {"type": "choice", "instructions": "Which team owns it?", "criteria": {"payments": null, "search": "search and listing"}},
		"severity": {"type": "score", "instructions": "How severe is it?", "criteria": ["minor", "major", "critical"]}
	}
}`

func sampleDecision(cost *float64) llm.DecisionResponse {
	usage := llm.NewDecisionUsage(120, 3, 123)
	if cost != nil {
		usage = llm.NewDecisionUsageWithCost(120, 3, 123, *cost, "USD")
	}
	return llm.NewDecisionResponse("jev-2026-09", map[string]llm.Answer{
		"is_urgent": llm.NewNoulAnswer(0.93),
		"team":      llm.NewChoiceAnswer("payments", map[string]float64{"payments": 0.9, "search": 0.1}, 0.8),
		"severity":  llm.NewScoreAnswer(1.7, map[string]string{"0": "minor", "1": "major", "2": "critical"}, map[string]float64{"0": 0.05, "1": 0.2, "2": 0.75}, 0.6),
	}, usage)
}

func postDecision(t *testing.T, server *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/decisions", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)
	return w
}

func TestHandleDecisions_AnswersAndReportsUsage(t *testing.T) {
	cost := 0.0004
	client := &mockDecisionClient{response: sampleDecision(&cost)}
	recorder := &decisionRecorder{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "acme/jev"}), WithHook(recorder))

	w := postDecision(t, server, decisionBody)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body: %s", w.Code, w.Body.String())
	}
	var resp struct {
		Model   string `json:"model"`
		Answers map[string]struct {
			Type       string   `json:"type"`
			Noul       *float64 `json:"noul"`
			Choice     *string  `json:"choice"`
			Score      *float64 `json:"score"`
			Confidence *float64 `json:"confidence"`
		} `json:"answers"`
		Usage map[string]any `json:"usage"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Model != "acme/jev" {
		t.Errorf("model = %q, want the name the client asked for", resp.Model)
	}
	if a := resp.Answers["is_urgent"]; a.Type != "noul" || a.Noul == nil || *a.Noul != 0.93 {
		t.Errorf("is_urgent = %+v", a)
	}
	if a := resp.Answers["team"]; a.Choice == nil || *a.Choice != "payments" || a.Confidence == nil {
		t.Errorf("team = %+v", a)
	}
	if a := resp.Answers["severity"]; a.Score == nil || *a.Score != 1.7 {
		t.Errorf("severity = %+v", a)
	}
	if _, leaked := resp.Usage["cost"]; leaked {
		t.Errorf("the upstream cost must not reach the client")
	}

	if recorder.req == nil || recorder.req.Type != RequestTypeDecision {
		t.Fatalf("the post-response hooks must see a decision request")
	}
	tokens := recorder.res.TokensUsed
	if tokens == nil || tokens.PromptTokens != 120 || tokens.CompletionTokens != 3 || tokens.TotalTokens != 123 {
		t.Fatalf("tokens = %+v", tokens)
	}
	if tokens.Cost == nil || *tokens.Cost != cost || tokens.CostCurrency != "USD" {
		t.Errorf("the upstream cost must reach the hooks, got %+v", tokens)
	}

	if _, ok := client.questions["team"].(llm.ChoiceQuestion); !ok {
		t.Errorf("questions = %#v", client.questions)
	}
	if state, ok := client.state.(json.RawMessage); !ok || !strings.Contains(string(state), "checkout page") {
		t.Errorf("the state must travel upstream unchanged, got %#v", client.state)
	}
}

// The resolvers return an llm.Client: a decorator must carry the capability.
func TestHandleDecisions_ThroughARetryingClient(t *testing.T) {
	client := &mockDecisionClient{response: sampleDecision(nil)}
	server := NewServer(WithHook(&resolverHook{client: retry.NewClient(client, time.Millisecond, 1), model: "jev"}))

	if w := postDecision(t, server, decisionBody); w.Code != http.StatusOK {
		t.Fatalf("status = %d, body: %s", w.Code, w.Body.String())
	}
}

func TestHandleDecisions_RefusesAModelThatCannotDecide(t *testing.T) {
	server := NewServer(WithHook(&resolverHook{client: &mockChatClient{}, model: "gpt-4"}))

	w := postDecision(t, server, decisionBody)

	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "does not support decisions") {
		t.Fatalf("status = %d, body: %s", w.Code, w.Body.String())
	}
}

func TestHandleDecisions_RefusesMalformedQuestions(t *testing.T) {
	client := &mockDecisionClient{response: sampleDecision(nil)}
	server := NewServer(WithHook(&resolverHook{client: client, model: "jev"}))

	for name, body := range map[string]string{
		"no state":        `{"model":"jev","questions":{"q":{"type":"noul","instructions":"ok?"}}}`,
		"no question":     `{"model":"jev","state":"s","questions":{}}`,
		"unknown type":    `{"model":"jev","state":"s","questions":{"q":{"type":"rank","instructions":"ok?"}}}`,
		"one score level": `{"model":"jev","state":"s","questions":{"q":{"type":"score","instructions":"ok?","criteria":["low"]}}}`,
		"choice as list":  `{"model":"jev","state":"s","questions":{"q":{"type":"choice","instructions":"ok?","criteria":["a","b"]}}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if w := postDecision(t, server, body); w.Code != http.StatusBadRequest {
				t.Errorf("status = %d, body: %s", w.Code, w.Body.String())
			}
		})
	}
	if client.questions != nil {
		t.Errorf("a malformed request must not reach the provider")
	}
}

// Without usage from the upstream, the hooks get no counts rather than zeros,
// which would read as a free call.
func TestHandleDecisions_WithoutUsage(t *testing.T) {
	client := &mockDecisionClient{response: llm.NewDecisionResponse("jev", map[string]llm.Answer{"q": llm.NewNoulAnswer(1)}, nil)}
	recorder := &decisionRecorder{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "jev"}), WithHook(recorder))

	w := postDecision(t, server, `{"model":"jev","state":"s","questions":{"q":{"type":"noul","instructions":"ok?"}}}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body: %s", w.Code, w.Body.String())
	}
	if recorder.res == nil || recorder.res.TokensUsed != nil {
		t.Errorf("TokensUsed must stay nil, got %+v", recorder.res)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if usage, ok := body["usage"]; ok {
		t.Errorf("the body must leave the usage out, got %s", usage)
	}
}

// bodyRewriter replaces the request body, as a hook that redacts the state
// would.
type bodyRewriter struct{ body string }

func (b *bodyRewriter) Name() string  { return "test.rewriter" }
func (b *bodyRewriter) Priority() int { return 0 }
func (b *bodyRewriter) PreRequest(_ context.Context, req *ProxyRequest) (*HookResult, error) {
	req.Body = json.RawMessage(b.body)
	return nil, nil
}

func TestHandleDecisions_SendsTheBodyTheHooksLeave(t *testing.T) {
	client := &mockDecisionClient{response: sampleDecision(nil)}
	rewriter := &bodyRewriter{body: `{"model":"jev","state":"redacted","questions":{"other":{"type":"noul","instructions":"still ok?"}}}`}
	server := NewServer(WithHook(rewriter), WithHook(&resolverHook{client: client, model: "jev"}))

	if w := postDecision(t, server, decisionBody); w.Code != http.StatusOK {
		t.Fatalf("status = %d, body: %s", w.Code, w.Body.String())
	}
	if state, _ := json.Marshal(client.state); string(state) != `"redacted"` {
		t.Errorf("state sent upstream = %s, want the hook's", state)
	}
	if _, ok := client.questions["other"]; !ok || len(client.questions) != 1 {
		t.Errorf("questions sent upstream = %v, want the hook's", client.questions)
	}

	rewriter.body = `{"model":"jev","state":null}`
	if w := postDecision(t, server, decisionBody); w.Code != http.StatusInternalServerError {
		t.Errorf("a hook breaking the body: status = %d, want 500", w.Code)
	}
}

// sessionDecisionClient records the session the context carried.
type sessionDecisionClient struct {
	mockDecisionClient
	sessionID string
}

func (s *sessionDecisionClient) Decision(ctx context.Context, state any, questions llm.Questions, funcs ...llm.DecisionOptionFunc) (llm.DecisionResponse, error) {
	s.sessionID = llm.SessionIDFromContext(ctx)
	return s.mockDecisionClient.Decision(ctx, state, questions, funcs...)
}

func TestHandleDecisions_CarriesTheSession(t *testing.T) {
	client := &sessionDecisionClient{mockDecisionClient: mockDecisionClient{response: sampleDecision(nil)}}
	recorder := &decisionRecorder{}
	server := NewServer(WithHook(&resolverHook{client: client, model: "jev"}), WithHook(recorder))

	req := httptest.NewRequest(http.MethodPost, "/decisions", bytes.NewBufferString(decisionBody))
	req.Header.Set("X-Session-Id", "sess-1")
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body: %s", w.Code, w.Body.String())
	}
	if recorder.req == nil || recorder.req.SessionID != "sess-1" {
		t.Errorf("hooks must see the session, got %+v", recorder.req)
	}
	if client.sessionID != "sess-1" {
		t.Errorf("session on the upstream context = %q, want sess-1", client.sessionID)
	}
}

// Each decorator carries the capability of the client it wraps, and only
// that: wrapping a chat-only client must not make it look able to decide.
func TestDecorators_CarryTheDecisionCapability(t *testing.T) {
	decorators := map[string]func(llm.Client) llm.Client{
		"retry":          func(c llm.Client) llm.Client { return retry.NewClient(c, time.Millisecond, 1) },
		"ratelimit":      func(c llm.Client) llm.Client { return ratelimit.NewClient(c) },
		"circuitbreaker": func(c llm.Client) llm.Client { return circuitbreaker.NewClient(c, 3, time.Second) },
	}
	questions := llm.Questions{"q": llm.NoulQuestion{Instructions: "ok?"}}
	for name, wrap := range decorators {
		t.Run(name, func(t *testing.T) {
			decider := wrap(&mockDecisionClient{response: sampleDecision(nil)})
			if !llm.SupportsDecision(decider) {
				t.Fatalf("the capability of the wrapped client is lost")
			}
			if _, err := llm.Decide(context.Background(), decider, "s", questions); err != nil {
				t.Fatalf("decide: %v", err)
			}

			chatOnly := wrap(&mockChatClient{})
			if llm.SupportsDecision(chatOnly) {
				t.Errorf("a chat-only client must not look able to decide")
			}
			if _, err := llm.Decide(context.Background(), chatOnly, "s", questions); !errors.Is(err, llm.ErrUnavailable) {
				t.Errorf("err = %v, want llm.ErrUnavailable", err)
			}
		})
	}
}

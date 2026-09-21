package typesafe

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/bornholm/genai/llm"
)

// The three question types must reach the wire in the documented shape:
// the API rejects anything else with a 422, and the criteria of each type
// have a different structure (object, map, ordered array).
func TestDecisionClient_EncodesQuestions(t *testing.T) {
	var got map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-key" {
			t.Errorf("Authorization = %q, want %q", auth, "Bearer test-key")
		}
		if r.URL.Path != evaluationPath {
			t.Errorf("path = %q, want %q", r.URL.Path, evaluationPath)
		}

		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &got); err != nil {
			t.Fatalf("decoding request: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"model":"jev-1.13.0","answers":{},"usage":{"input_tokens":1,"output_tokens":2}}`)
	}))
	defer server.Close()

	client := NewDecisionClient(server.URL+evaluationPath, "test-key", DefaultModel)

	_, err := client.Decision(context.Background(), "Help! My payouts have been failing for 3 days.", llm.Questions{
		"is_urgent": llm.NoulQuestion{
			Instructions: "Does this convey urgency?",
			True:         "Explicitly time-sensitive",
			False:        "No urgency expressed",
		},
		"department": llm.ChoiceQuestion{
			Instructions: "Which team should handle this?",
			Criteria:     map[string]any{"billing": "Payments, invoicing, refunds", "sales": nil},
		},
		"frustration": llm.ScoreQuestion{
			Instructions: "How frustrated is the customer?",
			Criteria:     []any{"Calm", "Frustrated", "Very angry"},
		},
	})
	if err != nil {
		t.Fatalf("Decision() = %v", err)
	}

	if got["state"] != "Help! My payouts have been failing for 3 days." {
		t.Errorf("state = %v", got["state"])
	}
	if got["model"] != DefaultModel {
		t.Errorf("model = %v, want %q", got["model"], DefaultModel)
	}

	questions, _ := got["questions"].(map[string]any)

	noul, _ := questions["is_urgent"].(map[string]any)
	if noul["type"] != string(llm.QuestionTypeNoul) {
		t.Errorf("is_urgent.type = %v", noul["type"])
	}
	wantNoulCriteria := map[string]any{"true": "Explicitly time-sensitive", "false": "No urgency expressed"}
	if !reflect.DeepEqual(noul["criteria"], wantNoulCriteria) {
		t.Errorf("is_urgent.criteria = %v, want %v", noul["criteria"], wantNoulCriteria)
	}

	choice, _ := questions["department"].(map[string]any)
	criteria, _ := choice["criteria"].(map[string]any)
	// A nil description means "this option needs no extra detail": the
	// option must survive as a null, not vanish from the set.
	if _, exists := criteria["sales"]; !exists {
		t.Errorf("department.criteria = %v, want a 'sales' entry", criteria)
	}
	if criteria["sales"] != nil {
		t.Errorf("department.criteria.sales = %v, want null", criteria["sales"])
	}

	score, _ := questions["frustration"].(map[string]any)
	wantLevels := []any{"Calm", "Frustrated", "Very angry"}
	if !reflect.DeepEqual(score["criteria"], wantLevels) {
		t.Errorf("frustration.criteria = %v, want %v (ordered)", score["criteria"], wantLevels)
	}
}

// A noul without rubric must not carry an empty criteria object: two blank
// descriptions do not mean the same thing as no rubric at all.
func TestDecisionClient_OmitsEmptyNoulCriteria(t *testing.T) {
	payload, err := encodeQuestion(llm.NoulQuestion{Instructions: "Urgent?"})
	if err != nil {
		t.Fatalf("encodeQuestion() = %v", err)
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshalling: %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("decoding: %v", err)
	}
	if _, exists := decoded["criteria"]; exists {
		t.Errorf("criteria = %v, want it omitted", decoded["criteria"])
	}
}

func TestDecisionClient_DecodesAnswers(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"model": "jev-1.13.0",
			"answers": {
				"is_urgent": {"type":"noul","noul":0.95},
				"department": {"type":"choice","choice":"billing","probabilities":{"billing":0.88,"technical":0.12,"sales":0.0},"confidence":0.81},
				"frustration": {"type":"score","score":1.05,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.0,"1":0.95,"2":0.05},"confidence":0.92}
			},
			"usage": {"input_tokens":296,"output_tokens":20}
		}`)
	}))
	defer server.Close()

	client := NewDecisionClient(server.URL, "", "")

	response, err := client.Decision(context.Background(), "state", llm.Questions{
		"is_urgent": llm.NoulQuestion{Instructions: "Urgent?"},
	})
	if err != nil {
		t.Fatalf("Decision() = %v", err)
	}

	if response.Model() != "jev-1.13.0" {
		t.Errorf("Model() = %q, want the resolved version", response.Model())
	}

	urgent, err := llm.AnswerOf[llm.NoulAnswer](response, "is_urgent")
	if err != nil {
		t.Fatalf("AnswerOf(is_urgent) = %v", err)
	}
	if urgent.Noul() != 0.95 {
		t.Errorf("Noul() = %v, want 0.95", urgent.Noul())
	}

	department, err := llm.AnswerOf[llm.ChoiceAnswer](response, "department")
	if err != nil {
		t.Fatalf("AnswerOf(department) = %v", err)
	}
	if department.Choice() != "billing" {
		t.Errorf("Choice() = %q, want billing", department.Choice())
	}
	if department.Confidence() != 0.81 {
		t.Errorf("Confidence() = %v, want 0.81", department.Confidence())
	}
	if department.Probabilities()["technical"] != 0.12 {
		t.Errorf("Probabilities() = %v", department.Probabilities())
	}

	frustration, err := llm.AnswerOf[llm.ScoreAnswer](response, "frustration")
	if err != nil {
		t.Fatalf("AnswerOf(frustration) = %v", err)
	}
	if frustration.Score() != 1.05 {
		t.Errorf("Score() = %v, want 1.05", frustration.Score())
	}
	if frustration.Legend()["1"] != "Frustrated" {
		t.Errorf("Legend() = %v", frustration.Legend())
	}

	// The native payload reports no total: computing it keeps callers from
	// reading a 0 as "nothing was billed".
	if total := response.Usage().TotalTokens(); total != 316 {
		t.Errorf("TotalTokens() = %d, want 316", total)
	}
}

// A gateway that states the cost must have it surface: billing inferred
// from token counts is a guess, this is not.
func TestDecisionClient_ReportsCost(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.5}},
			"usage":{"input_tokens":296,"output_tokens":20,"total_tokens":316,"cost":0.0000124}}`)
	}))
	defer server.Close()

	response, err := NewDecisionClient(server.URL, "", "").
		Decision(context.Background(), "state", llm.Questions{"q": llm.NoulQuestion{Instructions: "Urgent?"}})
	if err != nil {
		t.Fatalf("Decision() = %v", err)
	}

	reporting, ok := response.Usage().(llm.CostReportingUsage)
	if !ok {
		t.Fatal("usage does not report cost")
	}
	amount, currency, ok := reporting.Cost()
	if !ok || amount != 0.0000124 || currency != "USD" {
		t.Errorf("Cost() = %v, %q, %v; want 0.0000124, USD, true", amount, currency, ok)
	}
}

// 429 and 529 both ask for a backoff; the retry wrapper keys off
// llm.IsRetryable, so both must answer true.
func TestDecisionClient_RetryableErrors(t *testing.T) {
	for _, test := range []struct {
		status    int
		retryable bool
	}{
		{status: http.StatusTooManyRequests, retryable: true},
		{status: 529, retryable: true},
		{status: http.StatusUnprocessableEntity, retryable: false},
		{status: http.StatusUnauthorized, retryable: false},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(test.status)
			io.WriteString(w, `{"error":{"message":"nope"}}`)
		}))

		_, err := NewDecisionClient(server.URL, "", "").
			Decision(context.Background(), "state", llm.Questions{"q": llm.NoulQuestion{Instructions: "Urgent?"}})
		server.Close()

		if err == nil {
			t.Fatalf("HTTP %d: Decision() = nil, want an error", test.status)
		}
		if got := llm.IsRetryable(err); got != test.retryable {
			t.Errorf("HTTP %d: IsRetryable() = %v, want %v", test.status, got, test.retryable)
		}

		var httpErr *llm.HTTPError
		if !stderrors.As(err, &httpErr) {
			t.Fatalf("HTTP %d: error is not an *llm.HTTPError: %v", test.status, err)
		}
		if httpErr.Body != "nope" {
			t.Errorf("HTTP %d: body = %q, want the API message", test.status, httpErr.Body)
		}
	}
}

// A per-call model overrides the configured one: the same client serves
// several aliases.
func TestDecisionClient_ModelOverride(t *testing.T) {
	var got struct {
		Model string `json:"model"`
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		io.WriteString(w, `{"model":"jev-1.13.0","answers":{},"usage":{}}`)
	}))
	defer server.Close()

	_, err := NewDecisionClient(server.URL, "", "jev-latest").Decision(
		context.Background(), "state",
		llm.Questions{"q": llm.NoulQuestion{Instructions: "Urgent?"}},
		llm.WithDecisionModel("jev-1.13.0"),
	)
	if err != nil {
		t.Fatalf("Decision() = %v", err)
	}
	if got.Model != "jev-1.13.0" {
		t.Errorf("model = %q, want the per-call override", got.Model)
	}
}

// An invalid question must never reach the network.
func TestDecisionClient_ValidatesBeforeSending(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request sent despite invalid questions")
	}))
	defer server.Close()

	client := NewDecisionClient(server.URL, "", "")

	if _, err := client.Decision(context.Background(), "state", llm.Questions{}); err == nil {
		t.Error("Decision() with no question = nil, want an error")
	}
	if _, err := client.Decision(context.Background(), nil, llm.Questions{"q": llm.NoulQuestion{Instructions: "Urgent?"}}); err == nil {
		t.Error("Decision() with no state = nil, want an error")
	}
}

// NewDecisionClientForBaseURL derives the evaluation endpoint from the API
// root, with or without a trailing slash.
func TestNewDecisionClientForBaseURL(t *testing.T) {
	for baseURL, want := range map[string]string{
		"":                             DefaultBaseURL + evaluationPath,
		"https://api.typesafe.ai/v1":   "https://api.typesafe.ai/v1/systemone",
		"https://api.typesafe.ai/v1/":  "https://api.typesafe.ai/v1/systemone",
		"https://gateway.internal/v1/": "https://gateway.internal/v1/systemone",
	} {
		if got := NewDecisionClientForBaseURL(baseURL, "", "").endpoint; got != want {
			t.Errorf("baseURL %q: endpoint = %q, want %q", baseURL, got, want)
		}
	}
}

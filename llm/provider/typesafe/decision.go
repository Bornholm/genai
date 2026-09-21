// Package typesafe implements the TypeSafe "system one" evaluation API: it
// answers typed questions about a state with calibrated probabilities.
//
// The wire format is shared with the gateways that expose the same models —
// OpenRouter's decisions endpoint speaks it verbatim — so [DecisionClient]
// is written against a plain endpoint URL and reused there rather than
// duplicated.
package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/pkg/errors"

	"github.com/bornholm/genai/llm"
)

// DefaultBaseURL is the root of the public TypeSafe API.
const DefaultBaseURL = "https://api.typesafe.ai/v1"

// DefaultModel is TypeSafe's flagship alias; it resolves to a concrete
// version, which the response reports back.
const DefaultModel = "jev-latest"

// evaluationPath is appended to the base URL to reach the evaluation
// endpoint.
const evaluationPath = "/systemone"

// maxResponseSize caps what is read from a response body. Answers are a few
// kilobytes at most; the cap keeps a misrouted request from filling memory.
const maxResponseSize = 8 << 20

// DecisionClient evaluates a state against typed questions.
//
// No SDK covers this endpoint, so the HTTP client is written here against
// the documented payload.
type DecisionClient struct {
	httpClient *http.Client
	endpoint   string
	apiKey     string
	model      string
	header     http.Header
}

// ClientOptionFunc configures a [DecisionClient] at construction.
type ClientOptionFunc func(c *DecisionClient)

// WithHTTPClient uses the given HTTP client instead of [http.DefaultClient].
func WithHTTPClient(httpClient *http.Client) ClientOptionFunc {
	return func(c *DecisionClient) {
		if httpClient != nil {
			c.httpClient = httpClient
		}
	}
}

// WithHeader sets an extra header on every request, such as the attribution
// headers gateways use for their rankings.
func WithHeader(key, value string) ClientOptionFunc {
	return func(c *DecisionClient) {
		c.header.Set(key, value)
	}
}

// NewDecisionClient builds a client posting to endpoint, which must be the
// full URL of the evaluation endpoint. Use [NewDecisionClientForBaseURL] to
// derive it from an API root instead.
func NewDecisionClient(endpoint, apiKey, model string, funcs ...ClientOptionFunc) *DecisionClient {
	if model == "" {
		model = DefaultModel
	}

	client := &DecisionClient{
		httpClient: http.DefaultClient,
		endpoint:   endpoint,
		apiKey:     apiKey,
		model:      model,
		header:     http.Header{},
	}

	for _, fn := range funcs {
		fn(client)
	}

	return client
}

// NewDecisionClientForBaseURL builds a client for the TypeSafe API rooted at
// baseURL. An empty baseURL means the public service.
func NewDecisionClientForBaseURL(baseURL, apiKey, model string, funcs ...ClientOptionFunc) *DecisionClient {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return NewDecisionClient(strings.TrimSuffix(baseURL, "/")+evaluationPath, apiKey, model, funcs...)
}

// Decision implements [llm.DecisionClient].
func (c *DecisionClient) Decision(ctx context.Context, state any, questions llm.Questions, funcs ...llm.DecisionOptionFunc) (llm.DecisionResponse, error) {
	opts := llm.NewDecisionOptions(funcs...)

	if state == nil {
		return nil, llm.NewValidationError("state", "state is required")
	}
	if err := questions.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	model := c.model
	if opts.Model != "" {
		model = opts.Model
	}

	encoded, err := encodeQuestions(questions)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	body, err := json.Marshal(decisionRequest{
		State:     state,
		Model:     model,
		Questions: encoded,
	})
	if err != nil {
		return nil, errors.WithStack(err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, errors.WithStack(err)
	}

	for key, values := range c.header {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSize))
	if err != nil {
		return nil, errors.WithStack(err)
	}

	// 429 and 529 (overloaded) both ask for a backoff: RateLimitError tags
	// the first, and llm.IsRetryable already covers the 5xx range, so the
	// retry wrapper handles both without knowing this endpoint.
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, errors.WithStack(llm.RateLimitError(resp.StatusCode, errorMessage(raw)))
	}

	var parsed decisionResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, errors.Wrapf(err, "unexpected response (HTTP %d)", resp.StatusCode)
	}

	answers, err := decodeAnswers(parsed.Answers)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return llm.NewDecisionResponse(parsed.Model, answers, parsed.Usage.toUsage()), nil
}

var _ llm.DecisionClient = &DecisionClient{}

// errorMessage extracts the human-readable part of an error body, falling
// back to the body itself: validation failures name the offending field
// there and that detail is what makes the error actionable.
func errorMessage(raw []byte) string {
	var parsed struct {
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
		Detail json.RawMessage `json:"detail"`
	}
	if err := json.Unmarshal(raw, &parsed); err == nil {
		if parsed.Error != nil && parsed.Error.Message != "" {
			return parsed.Error.Message
		}
		if len(parsed.Detail) > 0 {
			return string(parsed.Detail)
		}
	}
	return string(raw)
}

type decisionRequest struct {
	State     any            `json:"state"`
	Model     string         `json:"model"`
	Questions map[string]any `json:"questions"`
}

type noulCriteria struct {
	True  any `json:"true,omitempty"`
	False any `json:"false,omitempty"`
}

type noulQuestion struct {
	Type         llm.QuestionType `json:"type"`
	Instructions any              `json:"instructions"`
	Criteria     *noulCriteria    `json:"criteria,omitempty"`
}

type choiceQuestion struct {
	Type         llm.QuestionType `json:"type"`
	Instructions any              `json:"instructions"`
	Criteria     map[string]any   `json:"criteria"`
}

type scoreQuestion struct {
	Type         llm.QuestionType `json:"type"`
	Instructions any              `json:"instructions"`
	Criteria     []any            `json:"criteria"`
}

func encodeQuestions(questions llm.Questions) (map[string]any, error) {
	encoded := make(map[string]any, len(questions))
	for id, question := range questions {
		payload, err := encodeQuestion(question)
		if err != nil {
			return nil, errors.Wrapf(err, "could not encode question '%s'", id)
		}
		encoded[id] = payload
	}
	return encoded, nil
}

func encodeQuestion(question llm.Question) (any, error) {
	switch q := question.(type) {
	case llm.NoulQuestion:
		payload := noulQuestion{Type: llm.QuestionTypeNoul, Instructions: q.Instructions}
		// The criteria object is optional, and an empty one would read as
		// two blank rubrics rather than as "no rubric".
		if q.True != nil || q.False != nil {
			payload.Criteria = &noulCriteria{True: q.True, False: q.False}
		}
		return payload, nil

	case llm.ChoiceQuestion:
		// A nil description is meaningful — "this option needs no extra
		// detail" — so the entry is kept with a null value.
		criteria := make(map[string]any, len(q.Criteria))
		for option, description := range q.Criteria {
			criteria[option] = description
		}
		return choiceQuestion{Type: llm.QuestionTypeChoice, Instructions: q.Instructions, Criteria: criteria}, nil

	case llm.ScoreQuestion:
		return scoreQuestion{Type: llm.QuestionTypeScore, Instructions: q.Instructions, Criteria: q.Criteria}, nil

	default:
		return nil, llm.NewValidationError("type", "unsupported question type")
	}
}

type decisionResponse struct {
	Model   string                   `json:"model"`
	Answers map[string]answerPayload `json:"answers"`
	Usage   usagePayload             `json:"usage"`
}

type usagePayload struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	// TotalTokens is not part of the native payload; gateways add it.
	TotalTokens int64 `json:"total_tokens"`
	// Cost is what the gateway charged, in USD. A pointer tells "not
	// reported" from "free".
	Cost *float64 `json:"cost"`
}

func (u usagePayload) toUsage() llm.DecisionUsage {
	total := u.TotalTokens
	if total == 0 {
		total = u.InputTokens + u.OutputTokens
	}
	if u.Cost != nil && *u.Cost >= 0 {
		return llm.NewDecisionUsageWithCost(u.InputTokens, u.OutputTokens, total, *u.Cost, "USD")
	}
	return llm.NewDecisionUsage(u.InputTokens, u.OutputTokens, total)
}

type answerPayload struct {
	Type          llm.QuestionType   `json:"type"`
	Noul          *float64           `json:"noul"`
	Choice        *string            `json:"choice"`
	Score         *float64           `json:"score"`
	Legend        map[string]string  `json:"legend"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
}

func decodeAnswers(payloads map[string]answerPayload) (map[string]llm.Answer, error) {
	answers := make(map[string]llm.Answer, len(payloads))
	for id, payload := range payloads {
		answer, err := decodeAnswer(payload)
		if err != nil {
			return nil, errors.Wrapf(err, "could not decode answer '%s'", id)
		}
		answers[id] = answer
	}
	return answers, nil
}

func decodeAnswer(payload answerPayload) (llm.Answer, error) {
	switch payload.Type {
	case llm.QuestionTypeNoul:
		if payload.Noul == nil {
			return nil, errors.New("noul answer carries no value")
		}
		return llm.NewNoulAnswer(*payload.Noul), nil

	case llm.QuestionTypeChoice:
		if payload.Choice == nil {
			return nil, errors.New("choice answer carries no option")
		}
		return llm.NewChoiceAnswer(*payload.Choice, payload.Probabilities, payload.Confidence), nil

	case llm.QuestionTypeScore:
		if payload.Score == nil {
			return nil, errors.New("score answer carries no value")
		}
		return llm.NewScoreAnswer(*payload.Score, payload.Legend, payload.Probabilities, payload.Confidence), nil

	default:
		return nil, errors.Errorf("unsupported answer type '%s'", payload.Type)
	}
}

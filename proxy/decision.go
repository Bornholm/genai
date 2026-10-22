package proxy

import (
	"encoding/json"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// The decision route speaks the TypeSafe "system one" wire format, which
// OpenRouter's decisions endpoint shares: a client of either reaches the
// proxy by pointing its endpoint URL at /decisions.

type decisionRequest struct {
	Model     string                      `json:"model"`
	State     json.RawMessage             `json:"state"`
	Questions map[string]decisionQuestion `json:"questions"`
}

type decisionQuestion struct {
	Type         llm.QuestionType `json:"type"`
	Instructions any              `json:"instructions"`
	Criteria     json.RawMessage  `json:"criteria"`
}

type decisionNoulCriteria struct {
	True  any `json:"true"`
	False any `json:"false"`
}

// ParseDecisionRequest reads a decision request body: the model, the state to
// evaluate, kept as raw JSON so that it travels upstream unchanged, and the
// questions. The questions are validated, so that a malformed one is refused
// before any hook runs.
func ParseDecisionRequest(body json.RawMessage) (model string, state json.RawMessage, questions llm.Questions, err error) {
	var req decisionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return "", nil, nil, errors.Wrap(err, "could not parse decision request")
	}

	if len(req.State) == 0 || string(req.State) == "null" {
		return "", nil, nil, errors.New("state is required")
	}

	questions = make(llm.Questions, len(req.Questions))
	for id, q := range req.Questions {
		question, err := parseDecisionQuestion(q)
		if err != nil {
			return "", nil, nil, errors.Wrapf(err, "invalid question '%s'", id)
		}
		questions[id] = question
	}
	if err := questions.Validate(); err != nil {
		return "", nil, nil, errors.WithStack(err)
	}

	return req.Model, req.State, questions, nil
}

func parseDecisionQuestion(q decisionQuestion) (llm.Question, error) {
	hasCriteria := len(q.Criteria) > 0 && string(q.Criteria) != "null"

	switch q.Type {
	case llm.QuestionTypeNoul:
		question := llm.NoulQuestion{Instructions: q.Instructions}
		if hasCriteria {
			var criteria decisionNoulCriteria
			if err := json.Unmarshal(q.Criteria, &criteria); err != nil {
				return nil, errors.Wrap(err, "noul criteria must be an object with true and false")
			}
			question.True, question.False = criteria.True, criteria.False
		}
		return question, nil

	case llm.QuestionTypeChoice:
		var criteria map[string]any
		if hasCriteria {
			if err := json.Unmarshal(q.Criteria, &criteria); err != nil {
				return nil, errors.Wrap(err, "choice criteria must map each option to its description")
			}
		}
		return llm.ChoiceQuestion{Instructions: q.Instructions, Criteria: criteria}, nil

	case llm.QuestionTypeScore:
		var criteria []any
		if hasCriteria {
			if err := json.Unmarshal(q.Criteria, &criteria); err != nil {
				return nil, errors.Wrap(err, "score criteria must list the levels, lowest first")
			}
		}
		return llm.ScoreQuestion{Instructions: q.Instructions, Criteria: criteria}, nil

	default:
		return nil, errors.Errorf("unsupported question type '%s'", q.Type)
	}
}

type decisionResponse struct {
	Model   string                    `json:"model"`
	Answers map[string]decisionAnswer `json:"answers"`
	Usage   *decisionUsage            `json:"usage,omitempty"`
}

type decisionAnswer struct {
	Type          llm.QuestionType   `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        *string            `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

type decisionUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}

// FormatDecisionResponse converts a llm.DecisionResponse to the wire format,
// under the model name the resolver returned, as the other routes do. The
// cost an upstream reports stays out of the body: it is the proxy's price,
// not the client's, and reaches the hooks through TokenUsage. The usage is
// left out when the upstream reported none, as it is for the hooks.
func FormatDecisionResponse(res llm.DecisionResponse, model string) any {
	answers := make(map[string]decisionAnswer, len(res.Answers()))
	for id, answer := range res.Answers() {
		// A question left unanswered, or answered with a type the wire
		// format has no fields for, is left out, as if the upstream had not
		// returned it.
		if formatted, ok := formatDecisionAnswer(answer); ok {
			answers[id] = formatted
		}
	}

	body := decisionResponse{Model: model, Answers: answers}
	if usage := res.Usage(); usage != nil {
		body.Usage = &decisionUsage{
			InputTokens:  usage.InputTokens(),
			OutputTokens: usage.OutputTokens(),
			TotalTokens:  usage.TotalTokens(),
		}
	}
	return body
}

func formatDecisionAnswer(answer llm.Answer) (decisionAnswer, bool) {
	switch a := answer.(type) {
	case llm.NoulAnswer:
		noul := a.Noul()
		return decisionAnswer{Type: llm.QuestionTypeNoul, Noul: &noul}, true
	case llm.ChoiceAnswer:
		choice, confidence := a.Choice(), a.Confidence()
		return decisionAnswer{Type: llm.QuestionTypeChoice, Choice: &choice, Probabilities: a.Probabilities(), Confidence: &confidence}, true
	case llm.ScoreAnswer:
		score, confidence := a.Score(), a.Confidence()
		return decisionAnswer{Type: llm.QuestionTypeScore, Score: &score, Legend: a.Legend(), Probabilities: a.Probabilities(), Confidence: &confidence}, true
	default:
		return decisionAnswer{}, false
	}
}

// decisionTokenUsage reports a decision's usage to the hooks, in the fields
// the other routes fill: input tokens as prompt tokens, output tokens as
// completion tokens, and the cost when the upstream states it. It is nil when
// the upstream reported no usage: zeroed counts would read as a free call.
func decisionTokenUsage(usage llm.DecisionUsage) *TokenUsage {
	if usage == nil {
		return nil
	}
	tokens := &TokenUsage{
		PromptTokens:     int(usage.InputTokens()),
		CompletionTokens: int(usage.OutputTokens()),
		TotalTokens:      int(usage.TotalTokens()),
	}
	if cr, ok := usage.(llm.CostReportingUsage); ok {
		if amount, currency, ok := cr.Cost(); ok {
			tokens.Cost = &amount
			tokens.CostCurrency = currency
		}
	}
	return tokens
}

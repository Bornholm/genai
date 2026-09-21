package llm_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

// A malformed question must be caught before the request leaves, and the
// error must name the offending id: a 422 from the API says which field is
// wrong, but only after a round trip the caller paid for.
func TestQuestions_Validate(t *testing.T) {
	for _, test := range []struct {
		name      string
		questions llm.Questions
		wantErr   string
	}{
		{
			name:      "empty map",
			questions: llm.Questions{},
			wantErr:   "at least one question",
		},
		{
			name:      "empty id",
			questions: llm.Questions{"": llm.NoulQuestion{Instructions: "Urgent?"}},
			wantErr:   "question ids cannot be empty",
		},
		{
			name:      "noul without instructions",
			questions: llm.Questions{"is_urgent": llm.NoulQuestion{}},
			wantErr:   "is_urgent",
		},
		{
			name:      "choice without options",
			questions: llm.Questions{"dept": llm.ChoiceQuestion{Instructions: "Which team?"}},
			wantErr:   "at least one option",
		},
		{
			name: "score with a single level",
			questions: llm.Questions{"frustration": llm.ScoreQuestion{
				Instructions: "How frustrated?",
				Criteria:     []any{"Calm"},
			}},
			wantErr: "between 2 and 10 levels",
		},
		{
			name: "valid",
			questions: llm.Questions{
				"is_urgent": llm.NoulQuestion{Instructions: "Urgent?", True: "Time-sensitive", False: "Not"},
				"dept":      llm.ChoiceQuestion{Instructions: "Which team?", Criteria: map[string]any{"billing": "Payments", "sales": nil}},
				"frustration": llm.ScoreQuestion{
					Instructions: "How frustrated?",
					Criteria:     []any{"Calm", "Frustrated", "Very angry"},
				},
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := test.questions.Validate()
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("Validate() = nil, want an error mentioning %q", test.wantErr)
			}
			if !strings.Contains(err.Error(), test.wantErr) {
				t.Errorf("Validate() = %v, want an error mentioning %q", err, test.wantErr)
			}
		})
	}
}

// A choice caps at 255 options; the 256th is rejected by the API.
func TestChoiceQuestion_OptionLimit(t *testing.T) {
	criteria := map[string]any{}
	for i := 0; i <= llm.MaxChoiceOptions; i++ {
		criteria[fmt.Sprintf("opt_%03d", i)] = nil
	}

	err := llm.ChoiceQuestion{Instructions: "Pick one", Criteria: criteria}.Validate()
	if err == nil {
		t.Fatalf("Validate() = nil, want an error for %d options", len(criteria))
	}
}

func TestAnswerOf(t *testing.T) {
	response := llm.NewDecisionResponse("jev-1.13.0", map[string]llm.Answer{
		"is_urgent": llm.NewNoulAnswer(0.95),
		"dept":      llm.NewChoiceAnswer("billing", map[string]float64{"billing": 0.88, "sales": 0.12}, 0.81),
	}, llm.NewDecisionUsage(296, 20, 316))

	urgent, err := llm.AnswerOf[llm.NoulAnswer](response, "is_urgent")
	if err != nil {
		t.Fatalf("AnswerOf = %v", err)
	}
	if urgent.Noul() != 0.95 {
		t.Errorf("Noul() = %v, want 0.95", urgent.Noul())
	}

	if _, err := llm.AnswerOf[llm.NoulAnswer](response, "missing"); !errors.Is(err, llm.ErrAnswerNotFound) {
		t.Errorf("AnswerOf(missing) = %v, want ErrAnswerNotFound", err)
	}

	// Asking for the wrong kind must fail loudly rather than hand back a
	// zero value that reads as "no" or "0".
	if _, err := llm.AnswerOf[llm.NoulAnswer](response, "dept"); err == nil {
		t.Error("AnswerOf(dept) as a noul = nil, want an error")
	}
}

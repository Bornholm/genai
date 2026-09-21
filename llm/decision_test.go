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
	// zero value that reads as "no" or "0" — and the message has to name
	// the type that was expected, not just the one that came back.
	_, err = llm.AnswerOf[llm.NoulAnswer](response, "dept")
	if err == nil {
		t.Fatal("AnswerOf(dept) as a noul = nil, want an error")
	}
	if !strings.Contains(err.Error(), "llm.NoulAnswer") {
		t.Errorf("AnswerOf(dept) = %v, want the expected type named", err)
	}
}

// The question types have value receivers, so their pointer forms satisfy
// llm.Question and compile — but no provider encodes them. They must be
// rejected here, where the id is still known, rather than late and
// unhelpfully.
func TestQuestions_RejectsPointerForms(t *testing.T) {
	err := llm.Questions{"q": &llm.NoulQuestion{Instructions: "Urgent?"}}.Validate()
	if err == nil {
		t.Fatal("Validate() with a pointer question = nil, want an error")
	}
	if !strings.Contains(err.Error(), "by value") {
		t.Errorf("Validate() = %v, want an error pointing at the value form", err)
	}
	if !strings.Contains(err.Error(), "q") {
		t.Errorf("Validate() = %v, want the question id named", err)
	}
}

// A typed nil pointer is not a nil interface: it slips past a plain nil
// check and panics inside the value receiver.
func TestQuestions_RejectsTypedNil(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Validate() panicked on a typed nil: %v", r)
		}
	}()

	if err := (llm.Questions{"q": (*llm.NoulQuestion)(nil)}).Validate(); err == nil {
		t.Error("Validate() with a typed nil = nil, want an error")
	}
}

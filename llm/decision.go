package llm

import (
	"context"
	"fmt"
	"reflect"

	"github.com/pkg/errors"
)

// DecisionClient evaluates a state against typed questions and returns
// calibrated probabilities rather than free text.
//
// Like [ImageGenerationClient], it is deliberately NOT a member of [Client]:
// adding a method there breaks every existing implementation. Callers
// discover the capability with a type assertion:
//
//	if decider, ok := client.(llm.DecisionClient); ok {
//	    // ...
//	}
//
// The shape follows the TypeSafe "system one" API: a state, a map of named
// questions, and one answer per question under the same names.
type DecisionClient interface {
	Decision(ctx context.Context, state any, questions Questions, funcs ...DecisionOptionFunc) (DecisionResponse, error)
}

// Questions maps an id you choose to the question asked under it. The
// matching answer comes back under the same id. Ids are not part of the
// evaluation itself, they only pair questions with answers.
type Questions map[string]Question

// Validate reports the first malformed question, naming its id.
func (q Questions) Validate() error {
	if len(q) == 0 {
		return NewValidationError("questions", "at least one question is required")
	}
	for id, question := range q {
		if id == "" {
			return NewValidationError("questions", "question ids cannot be empty")
		}
		if err := checkQuestionValue(id, question); err != nil {
			return err
		}
		if err := question.Validate(); err != nil {
			return errors.Wrapf(err, "invalid question '%s'", id)
		}
	}
	return nil
}

// checkQuestionValue rejects what the question types declare but providers
// cannot encode.
//
// The three question types carry value receivers, so their pointer forms
// satisfy [Question] too and &NoulQuestion{...} compiles — but a provider
// switches on the value types and would reject it late, with a message
// naming neither the problem nor the fix. Worse, a typed nil pointer is not
// a nil interface: it slips past a plain nil check and panics inside the
// value receiver. Both are caught here, where the id is still known.
func checkQuestionValue(id string, question Question) error {
	if question == nil {
		return NewValidationError("questions."+id, "question cannot be nil")
	}

	value := reflect.ValueOf(question)
	if value.Kind() != reflect.Ptr {
		return nil
	}
	if value.IsNil() {
		return NewValidationError("questions."+id, "question cannot be nil")
	}

	return NewValidationError("questions."+id, fmt.Sprintf("question must be passed by value, not as a %T", question))
}

// QuestionType discriminates the three kinds of question, and the answer
// each one yields.
type QuestionType string

const (
	QuestionTypeNoul   QuestionType = "noul"
	QuestionTypeChoice QuestionType = "choice"
	QuestionTypeScore  QuestionType = "score"
)

// MaxChoiceOptions is the largest number of options a [ChoiceQuestion] may
// define, per the TypeSafe API reference (docs.typesafe.ai), read 2026-09.
const MaxChoiceOptions = 255

// Score rubrics hold between MinScoreLevels and MaxScoreLevels levels, per
// the TypeSafe API reference (docs.typesafe.ai), read 2026-09.
//
// These bounds are enforced locally so a question the service would refuse
// never reaches it. They therefore have to follow the upstream limits: left
// behind, they refuse calls the service would accept.
const (
	MinScoreLevels = 2
	MaxScoreLevels = 10
)

// Question is one of [NoulQuestion], [ChoiceQuestion] or [ScoreQuestion].
//
// Instructions and criteria descriptions are typed as any because the API
// accepts a string, an object or an array: a long question carrying data it
// refers to is expressed as a map holding the question in one field and the
// data in the others, referenced by name in backticks.
type Question interface {
	QuestionType() QuestionType
	Validate() error
}

// NoulQuestion is a yes/no question. Its answer is the probability that the
// answer is yes.
type NoulQuestion struct {
	// Instructions is the yes/no question to evaluate.
	Instructions any
	// True describes what a yes (a value near 1) means. Optional.
	True any
	// False describes what a no (a value near 0) means. Optional.
	False any
}

// QuestionType implements [Question].
func (q NoulQuestion) QuestionType() QuestionType { return QuestionTypeNoul }

// Validate implements [Question].
func (q NoulQuestion) Validate() error {
	return validateInstructions(q.Instructions)
}

// ChoiceQuestion picks one option from the set defined by its criteria. Its
// answer carries the chosen option and the full probability distribution.
type ChoiceQuestion struct {
	// Instructions is what the model should decide.
	Instructions any
	// Criteria maps each option to a description of it. A nil value means
	// the option needs no extra detail.
	Criteria map[string]any
}

// QuestionType implements [Question].
func (q ChoiceQuestion) QuestionType() QuestionType { return QuestionTypeChoice }

// Validate implements [Question].
func (q ChoiceQuestion) Validate() error {
	if err := validateInstructions(q.Instructions); err != nil {
		return err
	}
	if len(q.Criteria) == 0 {
		return NewValidationError("criteria", "a choice requires at least one option")
	}
	if len(q.Criteria) > MaxChoiceOptions {
		return NewValidationError("criteria", fmt.Sprintf("a choice accepts at most %d options, got %d", MaxChoiceOptions, len(q.Criteria)))
	}
	for option := range q.Criteria {
		if option == "" {
			return NewValidationError("criteria", "option names cannot be empty")
		}
	}
	return nil
}

// ScoreQuestion rates the state along an ordered rubric. Its answer is a
// probability-weighted position across the levels, so it can land between
// two of them.
type ScoreQuestion struct {
	// Instructions is what the model should rate.
	Instructions any
	// Criteria is the ordered list of level descriptions, lowest first.
	Criteria []any
}

// QuestionType implements [Question].
func (q ScoreQuestion) QuestionType() QuestionType { return QuestionTypeScore }

// Validate implements [Question].
func (q ScoreQuestion) Validate() error {
	if err := validateInstructions(q.Instructions); err != nil {
		return err
	}
	if len(q.Criteria) < MinScoreLevels || len(q.Criteria) > MaxScoreLevels {
		return NewValidationError("criteria", fmt.Sprintf("a score requires between %d and %d levels, got %d", MinScoreLevels, MaxScoreLevels, len(q.Criteria)))
	}
	for i, level := range q.Criteria {
		if level == nil {
			return NewValidationError("criteria", fmt.Sprintf("level %d has no description", i))
		}
	}
	return nil
}

// validateInstructions accepts what the API accepts there, a string, an
// object or an array, and rejects the rest before the request leaves.
//
// Anything else reaches the service as a JSON scalar and comes back a 422,
// after a round trip the caller already paid for. A struct is allowed: it
// marshals to an object, which is the whole point of passing a question and
// the data it refers to together.
func validateInstructions(instructions any) error {
	const field = "instructions"

	if instructions == nil {
		return NewValidationError(field, "instructions are required")
	}

	value := reflect.ValueOf(instructions)
	for value.Kind() == reflect.Ptr || value.Kind() == reflect.Interface {
		if value.IsNil() {
			return NewValidationError(field, "instructions are required")
		}
		value = value.Elem()
	}

	switch value.Kind() {
	case reflect.String, reflect.Map, reflect.Slice, reflect.Array:
		if value.Len() == 0 {
			return NewValidationError(field, "instructions are required")
		}
	case reflect.Struct:
		// Marshals to an object; there is no length to check.
	default:
		return NewValidationError(field, fmt.Sprintf("instructions must be a string, an object or an array, not a %s", value.Kind()))
	}

	return nil
}

var (
	_ Question = NoulQuestion{}
	_ Question = ChoiceQuestion{}
	_ Question = ScoreQuestion{}
)

// DecisionOptions gathers the per-call options shared by decision providers.
type DecisionOptions struct {
	// Model overrides the model the provider was configured with. Empty
	// keeps the configured one.
	Model string
}

func NewDecisionOptions(funcs ...DecisionOptionFunc) *DecisionOptions {
	opts := &DecisionOptions{}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

type DecisionOptionFunc func(opts *DecisionOptions)

// WithDecisionModel evaluates this call with the given model instead of the
// configured one.
func WithDecisionModel(model string) DecisionOptionFunc {
	return func(opts *DecisionOptions) {
		opts.Model = model
	}
}

// DecisionResponse carries one answer per question.
type DecisionResponse interface {
	// Model is the model that performed the evaluation, as reported by the
	// provider: an alias such as "jev-latest" resolves to a concrete
	// version here.
	Model() string
	// Answers are keyed by the ids used in the questions.
	Answers() map[string]Answer
	// Usage may be nil when the provider reports no metrics.
	Usage() DecisionUsage
}

// Answer is one of [NoulAnswer], [ChoiceAnswer] or [ScoreAnswer], matching
// the type of the question it answers.
type Answer interface {
	AnswerType() QuestionType
}

// NoulAnswer answers a [NoulQuestion].
type NoulAnswer interface {
	Answer
	// Noul is the yes/no answer, from 0 (no) to 1 (yes).
	Noul() float64
}

// ChoiceAnswer answers a [ChoiceQuestion].
type ChoiceAnswer interface {
	Answer
	// Choice is the highest-probability option.
	Choice() string
	// Probabilities maps every option to its probability; they sum to 1.
	Probabilities() map[string]float64
	// Confidence is how certain the model is, derived from the
	// distribution, between 0 and 1.
	Confidence() float64
}

// ScoreAnswer answers a [ScoreQuestion].
type ScoreAnswer interface {
	Answer
	// Score is the probability-weighted position across the levels; it can
	// land between two of them.
	Score() float64
	// Legend maps each level index, as a string, to its description.
	Legend() map[string]string
	// Probabilities maps each level index, as a string, to its
	// probability; they sum to 1.
	Probabilities() map[string]float64
	// Confidence is how certain the model is, derived from the
	// distribution, between 0 and 1.
	Confidence() float64
}

// ErrAnswerNotFound is returned by [AnswerOf] when no answer carries the
// requested id.
var ErrAnswerNotFound = errors.New("answer not found")

// AnswerOf retrieves the answer stored under id and narrows it to the
// expected kind:
//
//	urgent, err := llm.AnswerOf[llm.NoulAnswer](response, "is_urgent")
//
// It returns [ErrAnswerNotFound] when the id is absent, and a validation
// error when the answer is of another kind than T.
func AnswerOf[T Answer](response DecisionResponse, id string) (T, error) {
	var zero T

	if response == nil {
		return zero, errors.WithStack(ErrAnswerNotFound)
	}

	answer, exists := response.Answers()[id]
	if !exists {
		return zero, errors.Wrapf(ErrAnswerNotFound, "no answer for question '%s'", id)
	}
	// The mismatch branch below calls AnswerType on the answer. A map
	// holding a nil under a live id would panic there instead of returning
	// the error this function exists to return. Providers in this repo
	// never store one, but the map comes from whoever built the response.
	if answer == nil {
		return zero, NewValidationError(id, "answer is nil")
	}

	typed, ok := answer.(T)
	if !ok {
		// %T on zero would print "<nil>": T is an interface here, so its
		// zero value carries no type. reflect names it instead.
		expected := reflect.TypeOf((*T)(nil)).Elem()
		return zero, NewValidationError(id, fmt.Sprintf("answer is a %s, not a %s", answer.AnswerType(), expected))
	}

	return typed, nil
}

type BaseNoulAnswer struct {
	noul float64
}

// AnswerType implements [Answer].
func (a *BaseNoulAnswer) AnswerType() QuestionType { return QuestionTypeNoul }

// Noul implements [NoulAnswer].
func (a *BaseNoulAnswer) Noul() float64 { return a.noul }

func NewNoulAnswer(noul float64) *BaseNoulAnswer {
	return &BaseNoulAnswer{noul: noul}
}

type BaseChoiceAnswer struct {
	choice        string
	probabilities map[string]float64
	confidence    float64
}

// AnswerType implements [Answer].
func (a *BaseChoiceAnswer) AnswerType() QuestionType { return QuestionTypeChoice }

// Choice implements [ChoiceAnswer].
func (a *BaseChoiceAnswer) Choice() string { return a.choice }

// Probabilities implements [ChoiceAnswer].
func (a *BaseChoiceAnswer) Probabilities() map[string]float64 { return a.probabilities }

// Confidence implements [ChoiceAnswer].
func (a *BaseChoiceAnswer) Confidence() float64 { return a.confidence }

func NewChoiceAnswer(choice string, probabilities map[string]float64, confidence float64) *BaseChoiceAnswer {
	return &BaseChoiceAnswer{choice: choice, probabilities: probabilities, confidence: confidence}
}

type BaseScoreAnswer struct {
	score         float64
	legend        map[string]string
	probabilities map[string]float64
	confidence    float64
}

// AnswerType implements [Answer].
func (a *BaseScoreAnswer) AnswerType() QuestionType { return QuestionTypeScore }

// Score implements [ScoreAnswer].
func (a *BaseScoreAnswer) Score() float64 { return a.score }

// Legend implements [ScoreAnswer].
func (a *BaseScoreAnswer) Legend() map[string]string { return a.legend }

// Probabilities implements [ScoreAnswer].
func (a *BaseScoreAnswer) Probabilities() map[string]float64 { return a.probabilities }

// Confidence implements [ScoreAnswer].
func (a *BaseScoreAnswer) Confidence() float64 { return a.confidence }

func NewScoreAnswer(score float64, legend map[string]string, probabilities map[string]float64, confidence float64) *BaseScoreAnswer {
	return &BaseScoreAnswer{score: score, legend: legend, probabilities: probabilities, confidence: confidence}
}

var (
	_ NoulAnswer   = &BaseNoulAnswer{}
	_ ChoiceAnswer = &BaseChoiceAnswer{}
	_ ScoreAnswer  = &BaseScoreAnswer{}
)

type DecisionUsage interface {
	InputTokens() int64
	OutputTokens() int64
	TotalTokens() int64
}

type BaseDecisionUsage struct {
	inputTokens  int64
	outputTokens int64
	totalTokens  int64
	cost         *float64
	costCurrency string
}

// InputTokens implements [DecisionUsage].
func (u *BaseDecisionUsage) InputTokens() int64 { return u.inputTokens }

// OutputTokens implements [DecisionUsage].
func (u *BaseDecisionUsage) OutputTokens() int64 { return u.outputTokens }

// TotalTokens implements [DecisionUsage].
func (u *BaseDecisionUsage) TotalTokens() int64 { return u.totalTokens }

func NewDecisionUsage(inputTokens, outputTokens, totalTokens int64) *BaseDecisionUsage {
	return &BaseDecisionUsage{inputTokens: inputTokens, outputTokens: outputTokens, totalTokens: totalTokens}
}

// NewDecisionUsageWithCost reports what the evaluation actually cost, as
// gateways such as OpenRouter state it. Read it back with
// [CostReportingUsage].
func NewDecisionUsageWithCost(inputTokens, outputTokens, totalTokens int64, cost float64, currency string) *BaseDecisionUsage {
	return &BaseDecisionUsage{
		inputTokens:  inputTokens,
		outputTokens: outputTokens,
		totalTokens:  totalTokens,
		cost:         &cost,
		costCurrency: currency,
	}
}

// Cost implements [CostReportingUsage].
func (u *BaseDecisionUsage) Cost() (amount float64, currency string, ok bool) {
	if u.cost == nil {
		return 0, "", false
	}
	return *u.cost, u.costCurrency, true
}

var (
	_ DecisionUsage      = &BaseDecisionUsage{}
	_ CostReportingUsage = &BaseDecisionUsage{}
)

type BaseDecisionResponse struct {
	model   string
	answers map[string]Answer
	usage   DecisionUsage
}

// Model implements [DecisionResponse].
func (r *BaseDecisionResponse) Model() string { return r.model }

// Answers implements [DecisionResponse].
func (r *BaseDecisionResponse) Answers() map[string]Answer { return r.answers }

// Usage implements [DecisionResponse].
func (r *BaseDecisionResponse) Usage() DecisionUsage { return r.usage }

func NewDecisionResponse(model string, answers map[string]Answer, usage DecisionUsage) *BaseDecisionResponse {
	return &BaseDecisionResponse{model: model, answers: answers, usage: usage}
}

var _ DecisionResponse = &BaseDecisionResponse{}

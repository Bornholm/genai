package provider_test

import (
	"context"
	"errors"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

type testDecisionOptions struct {
	Model string `env:"MODEL"`
}

type testDecisionClient struct {
	questions llm.Questions
}

func (c *testDecisionClient) Decision(ctx context.Context, state any, questions llm.Questions, funcs ...llm.DecisionOptionFunc) (llm.DecisionResponse, error) {
	c.questions = questions
	return llm.NewDecisionResponse("jev-1.13.0", map[string]llm.Answer{
		"is_urgent": llm.NewNoulAnswer(0.95),
	}, llm.NewDecisionUsage(296, 20, 316)), nil
}

// A client configured for decisions alone must be buildable: the capability
// does not require a chat completion provider alongside it.
func TestCreate_DecisionOnly(t *testing.T) {
	const testProvider provider.Name = "test-decision-provider"

	inner := &testDecisionClient{}
	provider.RegisterDecision(
		testProvider,
		func() *testDecisionOptions { return &testDecisionOptions{Model: "jev-latest"} },
		func(ctx context.Context, opts *testDecisionOptions) (llm.DecisionClient, error) {
			return inner, nil
		},
	)

	ctx := context.Background()
	client, err := provider.Create(ctx, provider.WithDecision(testProvider, testDecisionOptions{}))
	if err != nil {
		t.Fatalf("Create() = %v", err)
	}

	// The capability is not part of llm.Client: callers reach it with a
	// type assertion, which is the contract being checked here.
	decider, ok := any(client).(llm.DecisionClient)
	if !ok {
		t.Fatalf("client %T does not implement llm.DecisionClient", client)
	}

	response, err := decider.Decision(ctx, "Help! My payouts are failing.", llm.Questions{
		"is_urgent": llm.NoulQuestion{Instructions: "Does this convey urgency?"},
	})
	if err != nil {
		t.Fatalf("Decision() = %v", err)
	}
	if len(inner.questions) != 1 {
		t.Errorf("questions reaching the provider = %v, want one", inner.questions)
	}

	urgent, err := llm.AnswerOf[llm.NoulAnswer](response, "is_urgent")
	if err != nil {
		t.Fatalf("AnswerOf() = %v", err)
	}
	if urgent.Noul() != 0.95 {
		t.Errorf("Noul() = %v, want 0.95", urgent.Noul())
	}
}

// A client built without a decision provider must say so rather than panic.
func TestClient_DecisionUnavailable(t *testing.T) {
	client := provider.NewClient(nil, nil, nil)

	_, err := client.Decision(context.Background(), "state", llm.Questions{
		"q": llm.NoulQuestion{Instructions: "Urgent?"},
	})
	if !errors.Is(err, llm.ErrUnavailable) {
		t.Errorf("Decision() = %v, want ErrUnavailable", err)
	}
}

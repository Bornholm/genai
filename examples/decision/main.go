package main

import (
	"context"
	"errors"
	"flag"
	"log"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"

	_ "github.com/bornholm/genai/llm/provider/all"
	"github.com/bornholm/genai/llm/provider/env"
)

var (
	envFile string = ".env"
	state   string = "Help! My payouts have been failing for 3 days."
)

func init() {
	flag.StringVar(&envFile, "env-file", envFile, "client configuration environment file")
	flag.StringVar(&state, "state", state, "the content to evaluate")
}

func main() {
	flag.Parse()
	ctx := context.Background()

	client, err := provider.Create(ctx, env.With("GENAI_", envFile))
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	// Decisions are an optional capability: it is not part of llm.Client,
	// so it is reached with a type assertion.
	//
	// A client from provider.Create always satisfies the interface, even
	// with no decision provider configured, so this assertion only catches
	// a wrapped client: llm/retry, llm/ratelimit and llm/circuitbreaker do
	// not forward the capability. An unconfigured one fails at the call
	// below with llm.ErrUnavailable.
	decider, ok := client.(llm.DecisionClient)
	if !ok {
		log.Fatal("[FATAL] this client does not carry the decision capability")
	}

	res, err := decider.Decision(ctx, state, llm.Questions{
		"is_urgent": llm.NoulQuestion{
			Instructions: "Does this convey urgency?",
			True:         "Explicitly time-sensitive",
			False:        "No urgency expressed",
		},
		"department": llm.ChoiceQuestion{
			Instructions: "Which team should handle this?",
			Criteria: map[string]any{
				"billing":   "Payments, invoicing, refunds",
				"technical": "Bugs, outages, integrations",
				"sales":     "Pricing, upgrades, new accounts",
			},
		},
		"frustration": llm.ScoreQuestion{
			Instructions: "How frustrated is the customer?",
			Criteria:     []any{"Calm", "Frustrated", "Very angry"},
		},
	})
	if err != nil {
		if errors.Is(err, llm.ErrUnavailable) {
			log.Fatal("[FATAL] no decision provider configured, set GENAI_DECISION_PROVIDER")
		}
		log.Fatalf("[FATAL] %s", err)
	}

	log.Printf("[MODEL] %s", res.Model())

	urgent, err := llm.AnswerOf[llm.NoulAnswer](res, "is_urgent")
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}
	log.Printf("[URGENT] %.2f", urgent.Noul())

	department, err := llm.AnswerOf[llm.ChoiceAnswer](res, "department")
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}
	log.Printf("[DEPARTMENT] %s (confidence: %.2f, distribution: %v)", department.Choice(), department.Confidence(), department.Probabilities())

	frustration, err := llm.AnswerOf[llm.ScoreAnswer](res, "frustration")
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}
	log.Printf("[FRUSTRATION] %.2f on %v (confidence: %.2f)", frustration.Score(), frustration.Legend(), frustration.Confidence())

	if usage := res.Usage(); usage != nil {
		log.Printf("[USAGE] %d input tokens, %d output tokens", usage.InputTokens(), usage.OutputTokens())
	}
}

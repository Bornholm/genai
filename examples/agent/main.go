package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/agent/task"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/retry"
	"github.com/pkg/errors"

	_ "github.com/bornholm/genai/llm/provider/all"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create a LLM chat completion client
	client, err := provider.Create(ctx, provider.WithChatCompletionOptions(
		provider.ClientOptions{
			Provider: "mistral",
			BaseURL:  "https://api.mistral.ai/v1/",
			APIKey:   os.Getenv("MISTRAL_API_KEY"),
			Model:    "mistral-small-latest",
		},
	))
	if err != nil {
		log.Fatalf("%+v", errors.WithStack(err))
	}

	client = retry.Wrap(client, time.Second, 3)

	// Create a task agent and give him some location/meteo related tools
	taskAgent := agent.New(
		task.NewHandler(
			client,
			task.WithDefaultTools(
				llm.NewFuncTool(
					"get_french_address_location",
					"retrieve the coordinates for a french postal adress",
					map[string]any{
						"type": "object",
						"properties": map[string]any{
							"postal_address": map[string]any{
								"description": "the postal address of the location",
								"type":        "string",
							},
						},
						"required":              []string{"postal_address"},
						"additionnalProperties": false,
					},
					getFrenchLocation,
				),
				llm.NewFuncTool(
					"get_weather",
					"get the weather at the given location",
					map[string]any{
						"type": "object",
						"properties": map[string]any{
							"latitude": map[string]any{
								"description": "the latitude of the location",
								"type":        "number",
							},
							"longitude": map[string]any{
								"description": "the longitude of the location",
								"type":        "number",
							},
						},
						"required":              []string{"latitude", "longitude"},
						"additionnalProperties": false,
					},
					getWeather,
				),
			),
		),
	)

	// Start running the agent
	go func() {
		defer cancel()

		if err := taskAgent.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
			log.Fatalf("%+v", errors.WithStack(err))
		}
	}()

	// Prepare the task agent query
	query := agent.NewMessageEvent("How should i dress today in Dijon ?")

	log.Printf("--- Query\n%s\n", query.Message())

	// Send the query to the agent
	if err := taskAgent.In(query); err != nil {
		log.Fatalf("%+v", errors.WithStack(err))
	}

	for evt := range taskAgent.Output() {
		switch typ := evt.(type) {
		case task.ThoughtEvent:
			log.Printf("--- Thought (%s) #%d\n%s\n\n", typ.Type(), typ.Iteration(), typ.Thought())
		case task.ResultEvent:
			log.Printf("--- Result\n%s\n", typ.Result())
			// We exit when the agent has emitted its response
			cancel()
		}
	}

	<-ctx.Done()
}

package main

import (
	"context"
	"log"
	"log/slog"
	"time"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/agent/task"
	"github.com/bornholm/genai/examples/tool"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/retry"
	"github.com/pkg/errors"

	_ "github.com/bornholm/genai/llm/provider/all"
	"github.com/bornholm/genai/llm/provider/env"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// Create a LLM chat completion client
	client, err := provider.Create(ctx, env.With("GENAI_", ".env"))
	if err != nil {
		log.Fatalf("%+v", errors.WithStack(err))
	}

	slog.SetLogLoggerLevel(slog.LevelDebug)

	client = retry.Wrap(client, time.Second, 3)

	// Create a task agent and give him some location/meteo related tools
	taskAgent := agent.New(
		task.NewHandler(
			client,
			task.WithDefaultTools(
				tool.GetFrenchLocation,
				tool.GetWeather,
			),
		),
	)

	// Start running the agent
	if _, _, err := taskAgent.Start(ctx); err != nil {
		log.Fatalf("%+v", errors.WithStack(err))
	}

	query := "Comment devrais-je m'habiller aujourd'hui à Dijon ?"

	result, err := task.Do(ctx, taskAgent, query,
		task.WithOnThought(func(evt task.ThoughtEvent) error {
			log.Printf("--- Thought (%s) #%d\n%s\n\n", evt.Type(), evt.Iteration(), evt.Thought())
			return nil
		}),
	)
	if err != nil {
		log.Fatalf("%+v", errors.WithStack(err))
	}

	log.Printf("--- Result\n%s\n", result.Result())
}

package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"time"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/agent/loop"
	"github.com/bornholm/genai/examples/tool"
	"github.com/bornholm/genai/llm"
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

	client = retry.NewClient(client, time.Second, 3)

	// Build system prompt with tools
	tools := []llm.Tool{tool.GetFrenchLocation, tool.GetWeather}
	toolInfos := make([]loop.ToolInfo, len(tools))
	for i, t := range tools {
		toolInfos[i] = loop.ToolInfo{
			Name:        t.Name(),
			Description: t.Description(),
		}
	}

	systemPrompt, err := loop.DefaultSystemPrompt(toolInfos, "")
	if err != nil {
		log.Fatalf("%+v", errors.WithStack(err))
	}

	// Optional: enable reasoning tokens to improve multi-step tool-use quality.
	// The model will think through the problem before each tool call, and that
	// reasoning is automatically preserved across turns so it can pick up where
	// it left off after receiving tool results.
	//
	// Use llm.ReasoningEffortHigh / llm.ReasoningEffortMedium / etc. to tune cost vs quality,
	// or set MaxTokens directly for providers like Anthropic Claude:
	//
	//   reasoningOpts := &llm.ReasoningOptions{MaxTokens: llm.Ptr(8000)}
	//
	reasoningOpts := llm.NewReasoningOptions(llm.ReasoningEffortMedium)

	// Create a loop handler with tools and reasoning enabled
	handler, err := loop.NewHandler(
		loop.WithClient(client),
		loop.WithTools(tools...),
		loop.WithSystemPrompt(systemPrompt),
		loop.WithMaxIterations(5),
		loop.WithReasoningOptions(reasoningOpts),
	)
	if err != nil {
		log.Fatalf("%+v", errors.WithStack(err))
	}

	// Create runner
	runner := agent.NewRunner(handler)

	query := "Comment devrais-je m'habiller aujourd'hui à Dijon ?"

	// Run the agent
	var result string
	err = runner.Run(ctx, agent.NewInput(query), func(evt agent.Event) error {
		switch evt.Type() {
		case agent.EventTypeComplete:
			data := evt.Data().(*agent.CompleteData)
			result = data.Message
			fmt.Printf("--- Complete\n%s\n", data.Message)
		case agent.EventTypeToolCallStart:
			data := evt.Data().(*agent.ToolCallStartData)
			fmt.Printf("--- Tool Call: %s(%v)\n", data.Name, data.Parameters)
		case agent.EventTypeToolCallDone:
			data := evt.Data().(*agent.ToolCallDoneData)
			fmt.Printf("--- Tool Result: %s\n", data.Result)
		case agent.EventTypeTodoUpdated:
			data := evt.Data().(*agent.TodoUpdatedData)
			fmt.Printf("--- Todo Updated: %v\n", data.Items)
		case agent.EventTypeReasoning:
			data := evt.Data().(*agent.ReasoningData)
			// Truncate for display — reasoning can be very long
			preview := data.Reasoning
			if len(preview) > 200 {
				preview = preview[:200] + "…"
			}
			fmt.Printf("--- Reasoning (%d detail blocks): %s\n", len(data.ReasoningDetails), preview)
		case agent.EventTypeError:
			data := evt.Data().(*agent.ErrorData)
			fmt.Printf("--- Error: %s\n", data.Message)
		}
		return nil
	})

	if err != nil {
		log.Fatalf("%+v", errors.WithStack(err))
	}

	fmt.Printf("\n--- Final Result\n%s\n", result)
}

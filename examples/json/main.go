package main

import (
	"context"
	"log"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/invopop/jsonschema"

	// Imports client implementations

	_ "github.com/bornholm/genai/llm/provider/all"
	"github.com/bornholm/genai/llm/provider/env"
)

// TaskPlan represents the structured response we expect from the LLM
type TaskPlan struct {
	DailyPlan  string `json:"daily_plan" jsonschema:"description=Overall strategy for organizing the day"`
	Tasks      []Task `json:"tasks" jsonschema:"description=List of tasks to complete"`
	TotalTime  int    `json:"total_time_minutes" jsonschema:"description=Total estimated time for all tasks"`
	Motivation string `json:"motivation" jsonschema:"description=Motivational message to help stay focused"`
}

// Task represents an individual task in the plan
type Task struct {
	Name        string `json:"name" jsonschema:"required,description=Name of the task"`
	Priority    int    `json:"priority" jsonschema:"required,description=Priority level (1-3 where 1 is highest),minimum=1,maximum=3"`
	Duration    int    `json:"duration_minutes" jsonschema:"required,description=Estimated duration in minutes"`
	TimeSlot    string `json:"time_slot" jsonschema:"required,description=Recommended time slot (e.g. morning afternoon evening)"`
	Description string `json:"description" jsonschema:"required,description=Detailed description and tips for the task"`
}

// Response represents the expected LLM response
type Response struct {
	TaskPlan TaskPlan `json:"taskPlan" jsonschema:"required,description=The task plan"`
}

func main() {
	ctx := context.Background()

	// Create a client with chat completion implementation
	client, err := provider.Create(ctx, env.With("GENAI_", ".env"))
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	// Create our chat completion history
	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "You're a personal motivation assistant. You help me getting focused on the tasks i must do. Always respond with valid JSON according to the provided schema."),
		llm.NewMessage(llm.RoleUser, "Today, i must do the dishes, do my english class lesson and reach my daily steps target. How should i organize myself ?"),
	}

	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}

	// Generate JSON schema from the TaskPlan struct
	schema := reflector.Reflect(&Response{})

	// The chat completion options will now be validated before sending
	res, err := client.ChatCompletion(ctx,
		llm.WithMessages(messages...),
		llm.WithTemperature(0.7), // This will be validated to be between 0 and 2
		llm.WithJSONResponse(llm.NewResponseSchema(
			"task_plan",
			"A structured daily task organization plan",
			schema,
		)),
	)
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	log.Printf("[RAW RESPONSE] %s", res.Message().Content())

	// Parse the JSON response using llm.ParseJSON
	jsonResponses, err := llm.ParseJSON[Response](res.Message())
	if err != nil {
		log.Fatalf("[FATAL] Failed to parse JSON response: %s", err)
	}

	if len(jsonResponses) == 0 {
		log.Fatalf("[FATAL] No responses found")
	}

	// Use the first parsed found schema-matching response
	jsonRes := jsonResponses[0]

	log.Printf("[PLAN] %s", jsonRes.TaskPlan.DailyPlan)
	log.Printf("[TOTAL TIME] %d minutes", jsonRes.TaskPlan.TotalTime)
	log.Printf("[TASKS]")
	for i, task := range jsonRes.TaskPlan.Tasks {
		log.Printf("  %d. %s (Priority: %d, Duration: %d min, Time: %s)",
			i+1, task.Name, task.Priority, task.Duration, task.TimeSlot)
		log.Printf("     %s", task.Description)
	}
	log.Printf("[MOTIVATION] %s", jsonRes.TaskPlan.Motivation)
}

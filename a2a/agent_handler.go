package a2a

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/agent/loop"
	"github.com/bornholm/genai/llm"
	"github.com/google/uuid"
	"github.com/pkg/errors"
)

// ToolsProvider is a function that returns the current set of tools
type ToolsProvider func() []llm.Tool

// AgentTaskHandler wraps an agent loop configuration and implements TaskHandler.
type AgentTaskHandler struct {
	store         *TaskStore
	llmClient     llm.Client
	tools         []llm.Tool
	toolsProvider ToolsProvider
	systemPrompt  string
	loopOpts      []loop.OptionFunc
}

// AgentTaskHandlerOptionFunc is a function that configures the AgentTaskHandler
type AgentTaskHandlerOptionFunc func(*AgentTaskHandler)

// NewAgentTaskHandler creates a new AgentTaskHandler
func NewAgentTaskHandler(llmClient llm.Client, funcs ...AgentTaskHandlerOptionFunc) *AgentTaskHandler {
	h := &AgentTaskHandler{
		store:     NewTaskStore(),
		llmClient: llmClient,
		tools:     []llm.Tool{},
		loopOpts:  []loop.OptionFunc{},
	}
	for _, fn := range funcs {
		fn(h)
	}

	return h
}

// WithTools sets the tools available to the agent
func WithTools(tools ...llm.Tool) AgentTaskHandlerOptionFunc {
	return func(h *AgentTaskHandler) {
		h.tools = tools
	}
}

// WithToolsProvider sets a dynamic tools provider function
func WithToolsProvider(provider ToolsProvider) AgentTaskHandlerOptionFunc {
	return func(h *AgentTaskHandler) {
		h.toolsProvider = provider
	}
}

// WithSystemPrompt sets the system prompt for the agent
func WithSystemPrompt(prompt string) AgentTaskHandlerOptionFunc {
	return func(h *AgentTaskHandler) {
		h.systemPrompt = prompt
	}
}

// WithLoopOptions sets additional loop options
func WithLoopOptions(opts ...loop.OptionFunc) AgentTaskHandlerOptionFunc {
	return func(h *AgentTaskHandler) {
		h.loopOpts = append(h.loopOpts, opts...)
	}
}

// getTools returns the current set of tools, either from provider or static list
func (h *AgentTaskHandler) getTools() []llm.Tool {
	if h.toolsProvider != nil {
		return h.toolsProvider()
	}
	return h.tools
}

// createRunner creates a new agent runner with the current tools
func (h *AgentTaskHandler) createRunner() (*agent.Runner, error) {
	tools := h.getTools()

	// Create the loop handler
	handler, err := loop.NewHandler(
		append(h.loopOpts,
			loop.WithClient(h.llmClient),
			loop.WithTools(tools...),
			loop.WithSystemPrompt(h.systemPrompt),
		)...,
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create loop handler")
	}

	return agent.NewRunner(handler), nil
}

// HandleTask implements TaskHandler.HandleTask
func (h *AgentTaskHandler) HandleTask(ctx context.Context, params TaskSendParams) (*Task, error) {
	task := h.initTask(params)

	// Convert A2A message parts to the agent's user message
	userMessage := partsToText(params.Message.Parts)

	// Create runner with current tools
	runner, err := h.createRunner()
	if err != nil {
		task.Status = TaskStatus{
			State:     TaskStateFailed,
			Timestamp: time.Now(),
			Message:   &Message{Role: "agent", Parts: []Part{{Type: "text", Text: err.Error()}}},
		}
		h.store.Set(task)
		return task, nil
	}

	// Run the agent loop synchronously
	var finalContent string
	err = runner.Run(ctx, agent.NewInput(userMessage), func(evt agent.Event) error {
		// Capture the final response
		if evt.Type() == agent.EventTypeComplete {
			data := evt.Data().(*agent.CompleteData)
			finalContent = data.Message
		}
		return nil
	})
	if err != nil {
		task.Status = TaskStatus{
			State:     TaskStateFailed,
			Timestamp: time.Now(),
			Message:   &Message{Role: "agent", Parts: []Part{{Type: "text", Text: err.Error()}}},
		}
		h.store.Set(task)
		return task, nil // Return task in failed state, not an error
	}

	task.Status = TaskStatus{
		State:     TaskStateCompleted,
		Timestamp: time.Now(),
	}
	task.Artifacts = []Artifact{{
		Parts: []Part{{Type: "text", Text: finalContent}},
		Index: 0,
	}}
	h.store.Set(task)
	return task, nil
}

// HandleTaskSubscribe implements TaskHandler.HandleTaskSubscribe
func (h *AgentTaskHandler) HandleTaskSubscribe(ctx context.Context, params TaskSendParams, events chan<- any) error {
	defer close(events)

	task := h.initTask(params)

	// Send initial working status
	events <- TaskStatusUpdateEvent{
		ID: task.ID,
		Status: TaskStatus{
			State:     TaskStateWorking,
			Timestamp: time.Now(),
		},
		Final: false,
	}

	userMessage := partsToText(params.Message.Parts)

	// Create runner with current tools
	runner, err := h.createRunner()
	if err != nil {
		events <- TaskStatusUpdateEvent{
			ID: task.ID,
			Status: TaskStatus{
				State:     TaskStateFailed,
				Timestamp: time.Now(),
				Message:   &Message{Role: "agent", Parts: []Part{{Type: "text", Text: err.Error()}}},
			},
			Final: true,
		}
		return nil
	}

	var finalContent string
	err = runner.Run(ctx, agent.NewInput(userMessage), func(evt agent.Event) error {
		// Stream intermediate events
		switch evt.Type() {
		case agent.EventTypeToolCallStart:
			data := evt.Data().(*agent.ToolCallStartData)
			events <- TaskStatusUpdateEvent{
				ID: task.ID,
				Status: TaskStatus{
					State:     TaskStateWorking,
					Timestamp: time.Now(),
					Message: &Message{
						Role:  "agent",
						Parts: []Part{{Type: "text", Text: fmt.Sprintf("Calling tool: %s", data.Name)}},
					},
				},
				Final: false,
			}
		case agent.EventTypeToolCallDone:
			data := evt.Data().(*agent.ToolCallDoneData)
			events <- TaskStatusUpdateEvent{
				ID: task.ID,
				Status: TaskStatus{
					State:     TaskStateWorking,
					Timestamp: time.Now(),
					Message: &Message{
						Role:  "agent",
						Parts: []Part{{Type: "text", Text: fmt.Sprintf("Tool %s completed", data.Name)}},
					},
				},
				Final: false,
			}
		case agent.EventTypeComplete:
			data := evt.Data().(*agent.CompleteData)
			finalContent = data.Message
		}
		return nil
	})

	if err != nil {
		events <- TaskStatusUpdateEvent{
			ID: task.ID,
			Status: TaskStatus{
				State:     TaskStateFailed,
				Timestamp: time.Now(),
				Message:   &Message{Role: "agent", Parts: []Part{{Type: "text", Text: err.Error()}}},
			},
			Final: true,
		}
		return nil
	}

	// Send artifact
	events <- TaskArtifactUpdateEvent{
		ID: task.ID,
		Artifact: Artifact{
			Parts: []Part{{Type: "text", Text: finalContent}},
			Index: 0,
		},
	}

	// Send final completed status
	task.Status = TaskStatus{
		State:     TaskStateCompleted,
		Timestamp: time.Now(),
	}
	task.Artifacts = []Artifact{{
		Parts: []Part{{Type: "text", Text: finalContent}},
		Index: 0,
	}}
	h.store.Set(task)

	events <- TaskStatusUpdateEvent{
		ID:     task.ID,
		Status: task.Status,
		Final:  true,
	}

	return nil
}

// GetTask implements TaskHandler.GetTask
func (h *AgentTaskHandler) GetTask(ctx context.Context, params TaskQueryParams) (*Task, error) {
	task, ok := h.store.Get(params.ID)
	if !ok {
		return nil, fmt.Errorf("task not found: %s", params.ID)
	}
	// Apply historyLength truncation if requested
	if params.HistoryLength != nil && *params.HistoryLength < len(task.History) {
		task.History = task.History[len(task.History)-*params.HistoryLength:]
	}
	return task, nil
}

// CancelTask implements TaskHandler.CancelTask
func (h *AgentTaskHandler) CancelTask(ctx context.Context, params TaskQueryParams) (*Task, error) {
	task, ok := h.store.Get(params.ID)
	if !ok {
		return nil, fmt.Errorf("task not found: %s", params.ID)
	}
	task.Status = TaskStatus{
		State:     TaskStateCanceled,
		Timestamp: time.Now(),
	}
	h.store.Set(task)
	// NOTE: actual cancellation of the running goroutine requires
	// context cancellation, which should be wired at run time.
	return task, nil
}

// initTask creates and stores a new task from the given params
func (h *AgentTaskHandler) initTask(params TaskSendParams) *Task {
	taskID := params.ID
	if taskID == "" {
		taskID = uuid.New().String()
	}
	task := &Task{
		ID:        taskID,
		SessionID: params.SessionID,
		Status: TaskStatus{
			State:     TaskStateSubmitted,
			Timestamp: time.Now(),
		},
		History:  []Message{params.Message},
		Metadata: params.Metadata,
	}
	h.store.Set(task)
	return task
}

// partsToText extracts text from message parts
func partsToText(parts []Part) string {
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			sb.WriteString(p.Text)
			sb.WriteString("\n")
		}
	}
	return strings.TrimSpace(sb.String())
}

var _ TaskHandler = &AgentTaskHandler{}

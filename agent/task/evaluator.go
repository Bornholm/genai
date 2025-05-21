package task

import (
	"context"
	"log/slog"
	"strings"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/prompt"
	"github.com/pkg/errors"
)

type Evaluator interface {
	ShouldContinue(ctx context.Context, query string, response string, currentIteration, maxIterations int) (bool, string, error)
}

type LLMJudge struct {
	client  llm.ChatCompletionClient
	options []llm.ChatCompletionOptionFunc
}

// ShouldContinue implements Evaluator.
func (j *LLMJudge) ShouldContinue(ctx context.Context, query string, response string, currentIteration, maxIterations int) (bool, string, error) {
	systemPrompt, err := prompt.FromFS[any](&prompts, "prompts/judge_system.gotmpl", nil)
	if err != nil {
		return false, "", errors.WithStack(err)
	}

	tools := agent.ContextTools(ctx, []llm.Tool{})
	temperature := agent.ContextTemperature(ctx, 0.3)
	seed := agent.ContextSeed(ctx, -1)

	userPrompt, err := prompt.FromFS(&prompts, "prompts/judge_user.gotmpl", struct {
		Query            string
		CurrentIteration int
		MaxIterations    int
		Response         string
		Tools            []llm.Tool
	}{
		Query:            query,
		Response:         response,
		CurrentIteration: currentIteration,
		MaxIterations:    maxIterations,
		Tools:            tools,
	})
	if err != nil {
		return false, "", errors.WithStack(err)
	}

	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, systemPrompt),
		llm.NewMessage(llm.RoleUser, userPrompt),
	}

	options := []llm.ChatCompletionOptionFunc{
		llm.WithMessages(messages...),
		llm.WithTemperature(temperature),
	}

	if seed != -1 {
		options = append(options, llm.WithSeed(seed))
	}

	options = append(options, j.options...)

	res, err := j.client.ChatCompletion(ctx, options...)
	if err != nil {
		return false, "", errors.WithStack(err)
	}

	content := res.Message().Content()

	slog.DebugContext(ctx, "evaluator response", slog.String("response", content))

	if strings.Contains(content, "__STOP__") {
		return false, content, nil
	}

	return true, content, nil
}

var _ Evaluator = &LLMJudge{}

func NewLLMJudge(client llm.ChatCompletionClient, options ...llm.ChatCompletionOptionFunc) *LLMJudge {
	return &LLMJudge{
		client:  client,
		options: options,
	}
}

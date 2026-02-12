package task

import (
	"context"
	"embed"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/internal/logx"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/messageutil"
	"github.com/bornholm/genai/llm/prompt"
	"github.com/pkg/errors"
)

//go:embed prompts/*.gotmpl
var prompts embed.FS

// Default configuration constants
const (
	DefaultMinIterations = 1
	DefaultMaxIterations = 5
	DefaultTemperature   = 0.3
	DefaultSeed          = -1
)

type HandlerOptions struct {
	DefaultTools     []llm.Tool
	DefaultEvaluator Evaluator
}

type HandlerOptionFunc func(opts *HandlerOptions)

func WithDefaultTools(tools ...llm.Tool) HandlerOptionFunc {
	return func(opts *HandlerOptions) {
		opts.DefaultTools = tools
	}
}

func WithDefaultEvaluator(evaluator Evaluator) HandlerOptionFunc {
	return func(opts *HandlerOptions) {
		opts.DefaultEvaluator = evaluator
	}
}

func NewHandlerOptions(funcs ...HandlerOptionFunc) *HandlerOptions {
	opts := &HandlerOptions{
		DefaultTools:     []llm.Tool{},
		DefaultEvaluator: nil,
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

type Handler struct {
	defaultClient    llm.ChatCompletionClient
	defaultTools     []llm.Tool
	defaultEvaluator Evaluator
}

// Handle implements agent.Handler.
func (h *Handler) Handle(input agent.Event, outputs chan agent.Event) error {
	messageEvent, ok := input.(agent.MessageEvent)
	if !ok {
		return errors.Wrapf(agent.ErrNotSupported, "event type '%T' not supported", input)
	}

	ctx := input.Context()

	minIterations := ContextMinIterations(ctx, DefaultMinIterations)
	maxIterations := ContextMaxIterations(ctx, DefaultMaxIterations)
	client := agent.ContextClient(ctx, h.defaultClient)
	tools := agent.ContextTools(ctx, h.defaultTools)
	evaluator := ContextEvaluator(ctx, h.defaultEvaluator)
	messages := agent.ContextMessages(ctx, []llm.Message{})
	temperature := agent.ContextTemperature(ctx, DefaultTemperature)
	seed := agent.ContextSeed(ctx, DefaultSeed)
	additionalContext := ContextAdditionalContext(ctx, "")

	baseOptions := []llm.ChatCompletionOptionFunc{
		llm.WithTemperature(temperature),
	}
	if seed != DefaultSeed {
		baseOptions = append(baseOptions, llm.WithSeed(seed))
	}

	task := messageEvent.Message()

	slog.DebugContext(ctx, "received new message", slog.String("task", task))

	systemPrompt, err := prompt.FromFS[any](&prompts, "prompts/task_system.gotmpl", struct {
		AdditionalContext string
	}{
		AdditionalContext: additionalContext,
	})
	if err != nil {
		return errors.WithStack(err)
	}

	messages = messageutil.InjectSystemPrompt(messageutil.WithoutRoles(messages, llm.RoleSystem), systemPrompt)

	messages = append(messages,
		llm.NewMessage(llm.RoleUser, task),
	)

	var (
		synthesis llm.ChatCompletionResponse
	)

	var thoughts []string

	var finalEvaluation string

	for i := 0; i < maxIterations; i++ {
		slog.DebugContext(ctx, "new iteration", slog.Int("iteration", i))

		messages, err = h.next(ctx, client, tools, messages, baseOptions)
		if err != nil {
			return errors.WithStack(err)
		}

		iterationPrompt, err := prompt.FromFS[any](&prompts, "prompts/task_iteration.gotmpl", nil)
		if err != nil {
			return errors.WithStack(err)
		}

		messages = append(messages, llm.NewMessage(llm.RoleUser, iterationPrompt))

		res, err := client.ChatCompletion(ctx,
			llm.WithMessages(messages...),
			llm.WithToolChoice(llm.ToolChoiceNone),
			llm.WithTools(),
		)
		if err != nil {
			return errors.WithStack(err)
		}

		if thought := res.Message().Content(); thought != "" {
			messages = append(messages, res.Message())
			outputs <- NewThoughtEvent(ctx, i, ThoughtTypeAgent, thought, messageEvent)
			thoughts = append(thoughts, thought)
		}

		if i >= minIterations-1 {
			var wholeThoughts strings.Builder

			wholeThoughts.WriteString("## Tool calls\n\n")

			toolCallIndex := 0
			for _, m := range messages {
				if m.Role() != llm.RoleToolCalls {
					continue
				}

				toolCalls, ok := m.(llm.ToolCallsMessage)
				if !ok {
					continue
				}

				for _, c := range toolCalls.ToolCalls() {
					wholeThoughts.WriteString(strconv.FormatInt(int64(toolCallIndex), 10))
					wholeThoughts.WriteString(". ")
					wholeThoughts.WriteString(c.Name())
					wholeThoughts.WriteString("(" + fmt.Sprintf("%s", c.Parameters()) + ")\n")
					toolCallIndex++
				}
			}

			wholeThoughts.WriteString("\n## Thoughts\n\n")

			for i, t := range thoughts {
				wholeThoughts.WriteString("### Thought ")
				wholeThoughts.WriteString(strconv.FormatInt(int64(i), 10))
				wholeThoughts.WriteString("\n\n")
				wholeThoughts.WriteString(t)
				wholeThoughts.WriteString("\n\n")
			}

			evaluatorCtx := agent.WithContextTools(ctx, tools)

			shouldContinue, evaluatorThought, err := evaluator.ShouldContinue(evaluatorCtx, task, wholeThoughts.String(), i, maxIterations)
			if err != nil {
				return errors.WithStack(err)
			}

			slog.DebugContext(ctx, "evaluator judgement", slog.String("thought", evaluatorThought), slog.Bool("shouldContinue", shouldContinue))

			if evaluatorThought != "" {
				finalEvaluation = evaluatorThought
				outputs <- NewThoughtEvent(ctx, i, ThoughtTypeEvaluator, evaluatorThought, messageEvent)
			}

			if !shouldContinue {
				break
			}

			messages = append(messages, llm.NewMessage(llm.RoleSystem, "This is a independant analysis of your progress so far:\n\n"+evaluatorThought))
		}

	}

	synthetizeSystemPrompt, err := prompt.FromFS[any](&prompts, "prompts/task_synthetize_system.gotmpl", nil)
	if err != nil {
		return errors.WithStack(err)
	}

	synthetizeUserPrompt, err := prompt.FromFS(&prompts, "prompts/task_synthetize_user.gotmpl", struct {
		Task            string
		Thoughts        []string
		FinalEvaluation string
	}{
		Task:            task,
		Thoughts:        thoughts,
		FinalEvaluation: finalEvaluation,
	})
	if err != nil {
		return errors.WithStack(err)
	}

	messages = []llm.Message{
		llm.NewMessage(llm.RoleSystem, synthetizeSystemPrompt),
		llm.NewMessage(llm.RoleUser, synthetizeUserPrompt),
	}

	synthesisOptions := append(
		baseOptions,
		llm.WithMessages(messages...),
	)

	schema := ContextSchema(ctx, nil)
	if schema != nil {
		synthesisOptions = append(synthesisOptions, llm.WithJSONResponse(schema))
	}

	synthesis, err = client.ChatCompletion(ctx, synthesisOptions...)
	if err != nil {
		return errors.WithStack(err)
	}

	slog.DebugContext(ctx, "synthesis response", slog.String("synthesis", synthesis.Message().Content()))

	outputs <- NewResultEvent(ctx, synthesis.Message().Content(), thoughts, messageEvent)

	return nil
}

func (h *Handler) next(ctx context.Context, client llm.ChatCompletionClient, tools []llm.Tool, messages []llm.Message, baseOptions []llm.ChatCompletionOptionFunc) ([]llm.Message, error) {
	toolIteration := 0
	maxTooIterations := ContextMaxToolIterations(ctx, 5)

	for {
		iterationCtx := logx.WithAttrs(ctx, slog.Int("tool_iteration", toolIteration))

		options := append(
			baseOptions,
			llm.WithToolChoice(llm.ToolChoiceAuto),
			llm.WithMessages(messages...),
			llm.WithTools(tools...),
		)

		slog.DebugContext(iterationCtx, "executing iteration")

		res, err := client.ChatCompletion(ctx, options...)
		if err != nil {
			return nil, errors.WithStack(err)
		}

		hasToolCalls := len(res.ToolCalls()) > 0

		for _, tc := range res.ToolCalls() {
			tm, err := llm.ExecuteToolCall(ctx, tc, tools...)
			if err != nil {
				return nil, errors.WithStack(err)
			}

			messages = append(messages, tc, tm)
		}

		shouldEnd := toolIteration >= maxTooIterations

		if shouldEnd {
			return messages, nil
		}

		if hasToolCalls {
			toolIteration++
			continue
		} else {
			content := res.Message().Content()
			hasHallucinated := strings.Contains(content, "tool_call>") || strings.Contains(content, "TOOL_CALLS]") || strings.Contains(content, "<|tool_calls")
			if hasHallucinated {
				toolIteration++

				slog.WarnContext(iterationCtx, "model hallucinated tool call in text", slog.String("content", content))

				errorMsg := llm.NewMessage(llm.RoleSystem, "Error: You wrote a tool call in text. You must use the native Tool usage triggers provided by the API. Do not write XML or pseudo-code.")
				messages = append(messages, errorMsg)

				continue
			} else {
				return messages, nil
			}
		}
	}
}

func NewHandler(defaultClient llm.ChatCompletionClient, funcs ...HandlerOptionFunc) *Handler {
	opts := NewHandlerOptions(funcs...)

	if opts.DefaultEvaluator == nil {
		opts.DefaultEvaluator = NewLLMJudge(defaultClient)
	}

	return &Handler{
		defaultClient:    defaultClient,
		defaultEvaluator: opts.DefaultEvaluator,
		defaultTools:     opts.DefaultTools,
	}
}

var _ agent.Handler = &Handler{}

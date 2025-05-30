package simple

import (
	"context"
	"log/slog"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type HandlerOptions struct {
	DefaultTools []llm.Tool
}

type HandlerOptionFunc func(opts *HandlerOptions)

func WithDefaultTools(tools ...llm.Tool) HandlerOptionFunc {
	return func(opts *HandlerOptions) {
		opts.DefaultTools = tools
	}
}

func NewHandlerOptions(funcs ...HandlerOptionFunc) *HandlerOptions {
	opts := &HandlerOptions{
		DefaultTools: []llm.Tool{},
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

type Handler struct {
	defaultClient llm.ChatCompletionClient
	defaultTools  []llm.Tool
}

// Handle implements agent.Handler.
func (h *Handler) Handle(ctx context.Context, input agent.Event, outputs chan agent.Event) error {
	defer close(outputs)

	messageEvent, ok := input.(agent.MessageEvent)
	if !ok {
		return errors.Wrapf(agent.ErrNotSupported, "event type '%T' not supported", input)
	}

	client := agent.ContextClient(ctx, h.defaultClient)
	tools := agent.ContextTools(ctx, h.defaultTools)
	messages := agent.ContextMessages(ctx, []llm.Message{})
	temperature := agent.ContextTemperature(ctx, 0.3)
	seed := agent.ContextSeed(ctx, -1)

	message := messageEvent.Message()

	slog.DebugContext(ctx, "received new message", slog.String("message", message))

	messages = append(messages, llm.NewMessage(llm.RoleUser, message))

	toolChoice := llm.ToolChoiceAuto

	baseOptions := []llm.ChatCompletionOptionFunc{
		llm.WithTools(tools...),
		llm.WithTemperature(temperature),
	}

	if seed != -1 {
		baseOptions = append(baseOptions, llm.WithSeed(seed))
	}

	for {
		options := append(
			baseOptions,
			llm.WithToolChoice(toolChoice),
			llm.WithTools(tools...),
			llm.WithMessages(messages...),
		)

		res, err := client.ChatCompletion(ctx, options...)
		if err != nil {
			return errors.WithStack(err)
		}

		hasToolCalls := len(res.ToolCalls()) > 0

		for _, tc := range res.ToolCalls() {
			tm, err := llm.ExecuteToolCall(ctx, tc, tools...)
			if err != nil {
				return errors.WithStack(err)
			}

			messages = append(messages, tc, tm)
		}

		if hasToolCalls {
			toolChoice = llm.ToolChoiceNone
			continue
		}

		messages = append(messages, res.Message())

		break
	}

	outputs <- NewResponseEvent(messages[len(messages)-1].Content(), messageEvent)

	return nil
}

func NewHandler(defaultClient llm.ChatCompletionClient, funcs ...HandlerOptionFunc) *Handler {
	opts := NewHandlerOptions(funcs...)

	return &Handler{
		defaultClient: defaultClient,
		defaultTools:  opts.DefaultTools,
	}
}

var _ agent.Handler = &Handler{}

// Command testplugin is a deterministic in-memory provider used by the host
// side tests. Its behaviour is driven by options:
//
//	FAIL_WITH=rate_limit   every call fails with a 429
//	FAIL_WITH=validation   Configure fails with a validation error
//	HANG=true              streams block until the host cancels
//	NO_STREAM=true         the client does not implement streaming
//	PANIC=true             every chat completion call panics
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/plugin/sdk"
	"github.com/pkg/errors"
)

type options struct {
	Model    string `env:"MODEL"`
	FailWith string `env:"FAIL_WITH"`
	Hang     bool   `env:"HANG"`
	NoStream bool   `env:"NO_STREAM"`
	Panic    bool   `env:"PANIC"`
}

func (o *options) Validate() error {
	if o.FailWith == "validation" {
		return llm.NewValidationError("model", "model is required")
	}
	return nil
}

func main() {
	sdk.Serve(sdk.Config{
		Name:    "test",
		Version: "0.0.1",
		ChatCompletion: func(ctx context.Context, opts sdk.Options) (llm.ChatCompletionClient, error) {
			var o options
			if err := opts.Decode(&o); err != nil {
				return nil, err
			}
			if o.NoStream {
				return &completionOnlyClient{chatClient: &chatClient{opts: o}}, nil
			}
			return &chatClient{opts: o}, nil
		},
		Embeddings: func(ctx context.Context, opts sdk.Options) (llm.EmbeddingsClient, error) {
			var o options
			if err := opts.Decode(&o); err != nil {
				return nil, err
			}
			return &embeddingsClient{opts: o}, nil
		},
	})
}

type chatClient struct {
	opts options
}

func usage() llm.ChatCompletionUsage {
	cost := 0.5
	return llm.NewChatCompletionUsageFull(3, 5, 8, 1, 2, &cost, "USD")
}

func (c *chatClient) fail() error {
	if c.opts.FailWith == "rate_limit" {
		return llm.RateLimitError(429, "slow down")
	}
	return nil
}

func lastContent(opts *llm.ChatCompletionOptions) string {
	if len(opts.Messages) == 0 {
		return ""
	}
	return opts.Messages[len(opts.Messages)-1].Content()
}

func (c *chatClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	if c.opts.Panic {
		panic("provider bug")
	}
	if err := c.fail(); err != nil {
		return nil, err
	}
	opts := llm.NewChatCompletionOptions(funcs...)

	content := fmt.Sprintf("echo(%s): %s", c.opts.Model, lastContent(opts))
	if len(opts.Messages) > 0 && len(opts.Messages[0].Attachments()) > 0 {
		content += " +attachment"
	}

	var toolCalls []llm.ToolCall
	for _, tool := range opts.Tools {
		toolCalls = append(toolCalls, llm.NewToolCall("call-1", tool.Name(), `{"city":"Paris"}`))
	}

	message := llm.NewAssistantReasoningMessage(content, "thinking", nil)
	return llm.NewChatCompletionResponseWithReasoning(message, usage(), "thinking", nil, toolCalls...), nil
}

func (c *chatClient) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	opts := llm.NewChatCompletionOptions(funcs...)
	chunks := make(chan llm.StreamChunk, 10)

	go func() {
		defer close(chunks)

		if err := c.fail(); err != nil {
			llm.SendTerminalChunk(ctx, chunks, llm.NewErrorStreamChunk(err))
			return
		}

		if !llm.SendChunk(ctx, chunks, llm.NewStreamChunk(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", "thinking", nil))) {
			return
		}

		if c.opts.Hang {
			<-ctx.Done()
			llm.SendTerminalChunk(ctx, chunks, llm.NewErrorStreamChunk(errors.WithStack(ctx.Err())))
			return
		}

		content := fmt.Sprintf("echo(%s): %s", c.opts.Model, lastContent(opts))
		for _, word := range strings.SplitAfter(content, " ") {
			if !llm.SendChunk(ctx, chunks, llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, word))) {
				return
			}
		}

		for i, tool := range opts.Tools {
			if !llm.SendChunk(ctx, chunks, llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "", llm.NewToolCallDelta(i, "call-1", tool.Name(), `{"city":`)))) {
				return
			}
			if !llm.SendChunk(ctx, chunks, llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "", llm.NewToolCallDelta(i, "", "", `"Paris"}`)))) {
				return
			}
		}

		llm.SendTerminalChunk(ctx, chunks, llm.NewCompleteStreamChunk(usage()))
	}()

	return chunks, nil
}

// completionOnlyClient hides ChatCompletionStream, so that the SDK answers
// streaming requests from ChatCompletion.
type completionOnlyClient struct {
	chatClient *chatClient
}

func (c *completionOnlyClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return c.chatClient.ChatCompletion(ctx, funcs...)
}

type embeddingsClient struct {
	opts options
}

type embeddingsResponse struct {
	embeddings [][]float64
	usage      llm.EmbeddingsUsage
}

func (r *embeddingsResponse) Embeddings() [][]float64    { return r.embeddings }
func (r *embeddingsResponse) Usage() llm.EmbeddingsUsage { return r.usage }

func (c *embeddingsClient) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	if c.opts.FailWith == "rate_limit" {
		return nil, llm.RateLimitError(429, "slow down")
	}
	opts := llm.NewEmbeddingsOptions(funcs...)
	dims := 3
	if opts.Dimensions != nil {
		dims = *opts.Dimensions
	}
	res := &embeddingsResponse{usage: llm.NewEmbeddingsUsage(int64(len(inputs)), int64(len(inputs)))}
	for _, input := range inputs {
		vector := make([]float64, dims)
		for i := range vector {
			vector[i] = float64(len(input) + i)
		}
		res.embeddings = append(res.embeddings, vector)
	}
	return res, nil
}

package openai

import (
	"context"
	"io"
	"net/http"
	"sync/atomic"

	"github.com/bornholm/genai/llm"
	"github.com/davecgh/go-spew/spew"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/pkg/errors"
)

type ChatCompletionClient struct {
	client openai.Client
	params ParamsBuilder
}

// ChatCompletion implements llm.Client.
func (c *ChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	// Validate options before proceeding
	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	params, err := c.params.BuildParams(ctx, opts)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	var httpRes *http.Response

	completion, err := c.client.Chat.Completions.New(ctx, *params, option.WithResponseInto(&httpRes))
	if err != nil {
		if httpRes != nil && httpRes.StatusCode == http.StatusTooManyRequests {
			spew.Dump(httpRes.Header)

			return nil, errors.Wrap(llm.ErrRateLimit, err.Error())
		}

		if httpRes != nil && httpRes.Body != nil {

			body, readdErr := io.ReadAll(httpRes.Body)
			if readdErr != nil {
				return nil, errors.WithStack(err)
			}

			return nil, errors.Wrapf(err, "%s", body)
		}

		return nil, errors.WithStack(err)
	}

	if len(completion.Choices) == 0 {
		return nil, errors.WithStack(llm.ErrNoMessage)
	}

	openaiMessage := completion.Choices[0].Message

	var message llm.Message = llm.NewMessage(llm.RoleAssistant, openaiMessage.Content)

	toolCalls := make([]llm.ToolCall, 0)

	for _, tc := range openaiMessage.ToolCalls {
		toolCalls = append(toolCalls, llm.NewToolCall(tc.ID, tc.Function.Name, tc.Function.Arguments))
	}

	usage := llm.NewChatCompletionUsage(completion.Usage.PromptTokens, completion.Usage.CompletionTokens, completion.Usage.TotalTokens)

	return llm.NewChatCompletionResponse(message, usage, toolCalls...), nil
}

// ChatCompletionStream implements llm.ChatCompletionStreamingClient.
func (c *ChatCompletionClient) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	// Validate options before proceeding
	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	// Check if params builder supports streaming
	streamingBuilder, ok := c.params.(interface {
		BuildStreamingParams(ctx context.Context, opts *llm.ChatCompletionOptions) (*openai.ChatCompletionNewParams, error)
	})

	var params *openai.ChatCompletionNewParams
	var err error

	if ok {
		params, err = streamingBuilder.BuildStreamingParams(ctx, opts)
	} else {
		params, err = c.params.BuildParams(ctx, opts)
	}

	if err != nil {
		return nil, errors.WithStack(err)
	}

	var httpRes *http.Response

	stream := c.client.Chat.Completions.NewStreaming(ctx, *params, option.WithResponseInto(&httpRes))

	chunks := make(chan llm.StreamChunk, 10)

	var (
		promptTokens     atomic.Int64
		completionTokens atomic.Int64
		totalTokens      atomic.Int64
	)

	go func() {
		defer close(chunks)
		defer stream.Close()

		for stream.Next() {
			chunk := stream.Current()

			isNullUsage := chunk.Usage.CompletionTokens == 0 &&
				chunk.Usage.PromptTokens == 0 &&
				chunk.Usage.TotalTokens == 0

			if !isNullUsage {
				promptTokens.Store(chunk.Usage.PromptTokens)
				completionTokens.Store(chunk.Usage.CompletionTokens)
				totalTokens.Store(chunk.Usage.TotalTokens)
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			delta := choice.Delta

			// Create stream delta
			var toolCallDeltas []llm.ToolCallDelta
			for i, tc := range delta.ToolCalls {
				toolCallDeltas = append(toolCallDeltas, llm.NewToolCallDelta(
					i,
					tc.ID,
					tc.Function.Name,
					tc.Function.Arguments,
				))
			}

			streamDelta := llm.NewStreamDelta(
				llm.RoleAssistant,
				delta.Content,
				toolCallDeltas...,
			)

			chunks <- llm.NewStreamChunk(streamDelta)
		}

		if err := stream.Err(); err != nil {
			if httpRes != nil && httpRes.StatusCode == http.StatusTooManyRequests {
				chunks <- llm.NewErrorStreamChunk(errors.Wrap(llm.ErrRateLimit, err.Error()))
			} else {
				chunks <- llm.NewErrorStreamChunk(errors.WithStack(err))
			}
			return
		}

		// Send completion chunk
		chunks <- llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(
			promptTokens.Load(),
			completionTokens.Load(),
			totalTokens.Load(),
		))
	}()

	return chunks, nil
}

func NewChatCompletionClient(client openai.Client, params ParamsBuilder) *ChatCompletionClient {
	return &ChatCompletionClient{
		client: client,
		params: params,
	}
}

var _ llm.ChatCompletionClient = &ChatCompletionClient{}
var _ llm.ChatCompletionStreamingClient = &ChatCompletionClient{}

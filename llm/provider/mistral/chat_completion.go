package mistral

import (
	"context"
	"io"
	"net/http"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/pkg/errors"
)

// ChatCompletionClient is a Mistral-specific chat completion client
// that handles the structured content response with thinking blocks.
type ChatCompletionClient struct {
	client openai.Client
	params ParamsBuilder
}

// ParamsBuilder is the interface for building request parameters
type ParamsBuilder interface {
	BuildParams(ctx context.Context, opts *llm.ChatCompletionOptions) (*openai.ChatCompletionNewParams, error)
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

	// Extract thinking from response content
	textContent, reasoningDetails, reasoning := extractThinkingFromResponse(openaiMessage.Content)

	// Build the response message. When reasoning is present, return a ReasoningMessage
	// so callers can preserve it across turns.
	var message llm.Message
	if reasoning != "" || len(reasoningDetails) > 0 {
		message = llm.NewAssistantReasoningMessage(textContent, reasoning, reasoningDetails)
	} else {
		message = llm.NewMessage(llm.RoleAssistant, textContent)
	}

	toolCalls := make([]llm.ToolCall, 0)

	for _, tc := range openaiMessage.ToolCalls {
		toolCalls = append(toolCalls, llm.NewToolCall(tc.ID, tc.Function.Name, tc.Function.Arguments))
	}

	usage := llm.NewChatCompletionUsage(completion.Usage.PromptTokens, completion.Usage.CompletionTokens, completion.Usage.TotalTokens)

	// Return response with reasoning if present
	if reasoning != "" || len(reasoningDetails) > 0 {
		return llm.NewChatCompletionResponseWithReasoning(message, usage, reasoning, reasoningDetails, toolCalls...), nil
	}

	return llm.NewChatCompletionResponse(message, usage, toolCalls...), nil
}

// ChatCompletionStream implements llm.ChatCompletionStreamingClient.
func (c *ChatCompletionClient) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
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

	stream := c.client.Chat.Completions.NewStreaming(ctx, *params, option.WithResponseInto(&httpRes))

	chunks := make(chan llm.StreamChunk, 10)

	go func() {
		defer close(chunks)
		defer stream.Close()

		for stream.Next() {
			chunk := stream.Current()

			isNullUsage := chunk.Usage.CompletionTokens == 0 &&
				chunk.Usage.PromptTokens == 0 &&
				chunk.Usage.TotalTokens == 0

			if !isNullUsage {
				chunks <- llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(
					int64(chunk.Usage.PromptTokens),
					int64(chunk.Usage.CompletionTokens),
					int64(chunk.Usage.TotalTokens),
				))
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

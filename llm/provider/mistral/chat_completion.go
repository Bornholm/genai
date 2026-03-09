package mistral

import (
	"context"
	"io"
	"net/http"
	"strings"

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
		if httpRes != nil {
			body, _ := io.ReadAll(httpRes.Body)
			return nil, errors.WithStack(llm.NewHTTPError(httpRes.StatusCode, string(body)))
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
//
// Mistral may send its content as a raw JSON array containing thinking blocks
// (e.g. [{"type":"thinking",...},{"type":"text","text":"..."}]). To avoid
// leaking that raw JSON to callers, we buffer all delta.Content chunks and
// process them through extractThinkingFromResponse at the end of the stream,
// emitting a single properly-typed delta. Tool call deltas are forwarded
// individually as they arrive so accumulators in the caller stay correct.
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

		var (
			textBuf        strings.Builder
			reasoningBuf   strings.Builder
			reasoningDets  []llm.ReasoningDetail
			finalUsage     llm.ChatCompletionUsage
		)

		for stream.Next() {
			chunk := stream.Current()

			isNullUsage := chunk.Usage.CompletionTokens == 0 &&
				chunk.Usage.PromptTokens == 0 &&
				chunk.Usage.TotalTokens == 0

			// Save usage for later — emitting CompleteStreamChunk here would cause
			// doStreamingLLMCall to break before the content chunk is sent below.
			if !isNullUsage {
				finalUsage = llm.NewChatCompletionUsage(
					int64(chunk.Usage.PromptTokens),
					int64(chunk.Usage.CompletionTokens),
					int64(chunk.Usage.TotalTokens),
				)
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			delta := choice.Delta

			// Each delta.Content from Mistral is either a plain string or a
			// self-contained JSON array like [{"type":"thinking",...}].
			// Process each chunk individually so thinking blocks are stripped.
			if delta.Content != "" {
				chunkText, chunkDetails, chunkReasoning := extractThinkingFromResponse(delta.Content)
				if chunkReasoning != "" || len(chunkDetails) > 0 {
					reasoningBuf.WriteString(chunkReasoning)
					reasoningDets = append(reasoningDets, chunkDetails...)
				}
				if chunkText != "" {
					textBuf.WriteString(chunkText)
				}
			}

			// Tool call deltas are structured and safe to forward immediately.
			if len(delta.ToolCalls) > 0 {
				var toolCallDeltas []llm.ToolCallDelta
				for i, tc := range delta.ToolCalls {
					toolCallDeltas = append(toolCallDeltas, llm.NewToolCallDelta(
						i,
						tc.ID,
						tc.Function.Name,
						tc.Function.Arguments,
					))
				}
				chunks <- llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "", toolCallDeltas...))
			}
		}

		if err := stream.Err(); err != nil {
			if httpRes != nil {
				body, _ := io.ReadAll(httpRes.Body)
				chunks <- llm.NewErrorStreamChunk(errors.WithStack(llm.NewHTTPError(httpRes.StatusCode, string(body))))
			} else {
				chunks <- llm.NewErrorStreamChunk(errors.WithStack(err))
			}
			return
		}

		// Emit accumulated content/reasoning before the CompleteStreamChunk so
		// the handler does not break out of its receive loop before seeing content.
		reasoning := reasoningBuf.String()
		textContent := textBuf.String()
		if reasoning != "" || len(reasoningDets) > 0 {
			chunks <- llm.NewStreamChunk(llm.NewReasoningStreamDelta(
				llm.RoleAssistant,
				textContent,
				reasoning,
				reasoningDets,
			))
		} else if textContent != "" {
			chunks <- llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, textContent))
		}

		// Emit the complete (usage) chunk last so the handler breaks only after
		// all content and tool call deltas have been consumed.
		if finalUsage != nil {
			chunks <- llm.NewCompleteStreamChunk(finalUsage)
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

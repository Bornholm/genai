package openai

import (
	"context"
	"net/http"

	"github.com/bornholm/genai/llm"
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

	params, err := c.params.BuildParams(ctx, opts)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	var httpRes *http.Response

	completion, err := c.client.Chat.Completions.New(ctx, *params, option.WithResponseInto(&httpRes))
	if err != nil {
		if httpRes.StatusCode == http.StatusTooManyRequests {
			return nil, errors.Wrap(llm.ErrRateLimit, err.Error())
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

func NewChatCompletionClient(client openai.Client, params ParamsBuilder) *ChatCompletionClient {
	return &ChatCompletionClient{
		client: client,
		params: params,
	}
}

var _ llm.ChatCompletionClient = &ChatCompletionClient{}

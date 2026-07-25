package provider

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type Client struct {
	chatCompletion llm.ChatCompletionClient
	embeddings     llm.EmbeddingsClient
	transcription  llm.TranscriptionClient
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	if c.chatCompletion == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	response, err := c.chatCompletion.ChatCompletion(ctx, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return response, nil
}

// ChatCompletionStream implements llm.Client.
func (c *Client) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	streamingClient, ok := c.chatCompletion.(llm.ChatCompletionStreamingClient)
	if !ok {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	stream, err := streamingClient.ChatCompletionStream(ctx, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return stream, nil
}

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	if c.embeddings == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	response, err := c.embeddings.Embeddings(ctx, inputs, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return response, nil
}

// Transcription implements llm.Client.
func (c *Client) Transcription(ctx context.Context, audio []byte, funcs ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	if c.transcription == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	response, err := c.transcription.Transcription(ctx, audio, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return response, nil
}

func NewClient(chatCompletion llm.ChatCompletionClient, embeddings llm.EmbeddingsClient, transcription llm.TranscriptionClient) *Client {
	return &Client{
		chatCompletion: chatCompletion,
		embeddings:     embeddings,
		transcription:  transcription,
	}
}

var _ llm.Client = &Client{}

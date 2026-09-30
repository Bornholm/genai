package provider

import (
	"context"
	"io"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type Client struct {
	chatCompletion  llm.ChatCompletionClient
	embeddings      llm.EmbeddingsClient
	transcription   llm.TranscriptionClient
	imageGeneration llm.ImageGenerationClient
	decision        llm.DecisionClient
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

// ImageGeneration implements [llm.ImageGenerationClient].
//
// That interface is deliberately NOT part of [llm.Client]: adding a method
// there breaks every existing implementation. Callers reach it with a type
// assertion on the concrete client returned by Create.
func (c *Client) ImageGeneration(ctx context.Context, prompt string, funcs ...llm.ImageGenerationOptionFunc) (llm.ImageGenerationResponse, error) {
	if c.imageGeneration == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	response, err := c.imageGeneration.ImageGeneration(ctx, prompt, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return response, nil
}

// Decision implements [llm.DecisionClient].
//
// Like ImageGeneration, that interface is deliberately NOT part of
// [llm.Client]: callers reach it with a type assertion on the concrete
// client returned by Create.
func (c *Client) Decision(ctx context.Context, state any, questions llm.Questions, funcs ...llm.DecisionOptionFunc) (llm.DecisionResponse, error) {
	if c.decision == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	response, err := c.decision.Decision(ctx, state, questions, funcs...)
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

// NewClientWithImageGeneration builds a client that also generates images.
// Kept separate from NewClient so existing callers keep compiling.
func NewClientWithImageGeneration(chatCompletion llm.ChatCompletionClient, embeddings llm.EmbeddingsClient, transcription llm.TranscriptionClient, imageGeneration llm.ImageGenerationClient) *Client {
	client := NewClient(chatCompletion, embeddings, transcription)
	client.imageGeneration = imageGeneration
	return client
}

// NewClientWithDecision builds a client that also answers typed questions.
// Kept separate from NewClientWithImageGeneration so existing callers keep
// compiling.
func NewClientWithDecision(chatCompletion llm.ChatCompletionClient, embeddings llm.EmbeddingsClient, transcription llm.TranscriptionClient, imageGeneration llm.ImageGenerationClient, decision llm.DecisionClient) *Client {
	client := NewClientWithImageGeneration(chatCompletion, embeddings, transcription, imageGeneration)
	client.decision = decision
	return client
}

// Close releases the underlying clients that implement io.Closer, such as
// plugin clients holding a configured instance in a plugin process. Clients
// without a Close are left alone. Every closer is called even when one
// fails; the first error is returned. A client used for several capabilities
// gets closed once per capability, so its Close must tolerate repeats.
func (c *Client) Close() error {
	var first error
	for _, sub := range []any{c.chatCompletion, c.embeddings, c.transcription, c.imageGeneration, c.decision} {
		closer, ok := sub.(io.Closer)
		if !ok {
			continue
		}
		if err := closer.Close(); err != nil && first == nil {
			first = errors.WithStack(err)
		}
	}
	return first
}

var (
	_ llm.Client                = &Client{}
	_ llm.ImageGenerationClient = &Client{}
	_ llm.DecisionClient        = &Client{}
	_ io.Closer                 = &Client{}
)

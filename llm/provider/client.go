package provider

import (
	"context"
	"io"
	"net/http"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type Client struct {
	chatCompletion  llm.ChatCompletionClient
	embeddings      llm.EmbeddingsClient
	transcription   llm.TranscriptionClient
	imageGeneration llm.ImageGenerationClient
	decision        llm.DecisionClient
	discovery       llm.DiscoveryClient
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	if c.chatCompletion == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	response, err := c.chatCompletion.ChatCompletion(withSessionID(ctx, funcs), funcs...)
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

	stream, err := streamingClient.ChatCompletionStream(withSessionID(ctx, funcs), funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return stream, nil
}

// withSessionID puts the SessionID option on the context, where the HTTP
// client of a provider configured with a session header reads it.
func withSessionID(ctx context.Context, funcs []llm.ChatCompletionOptionFunc) context.Context {
	return llm.ContextWithSessionID(ctx, llm.NewChatCompletionOptions(funcs...).SessionID)
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

// SupportsDecision reports whether a decision client was configured, see
// [llm.SupportsDecision]: Decision exists whatever the configuration.
func (c *Client) SupportsDecision() bool {
	return c.decision != nil
}

// Models implements [llm.DiscoveryClient].
//
// Like ImageGeneration, that interface is deliberately NOT part of
// [llm.Client]: callers reach it with a type assertion on the concrete
// client returned by Create.
func (c *Client) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	if c.discovery == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	models, err := c.discovery.Models(ctx)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return models, nil
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

// NewClientWithDiscovery builds a client that also lists the provider's
// models. Kept separate from NewClientWithDecision so existing callers keep
// compiling.
func NewClientWithDiscovery(chatCompletion llm.ChatCompletionClient, embeddings llm.EmbeddingsClient, transcription llm.TranscriptionClient, imageGeneration llm.ImageGenerationClient, decision llm.DecisionClient, discovery llm.DiscoveryClient) *Client {
	client := NewClientWithDecision(chatCompletion, embeddings, transcription, imageGeneration, decision)
	client.discovery = discovery
	return client
}

// Close releases the underlying clients that implement io.Closer, such as
// plugin clients holding a configured instance in a plugin process. Clients
// without a Close are left alone. Every closer is called even when one
// fails; the first error is returned. A client used for several capabilities
// gets closed once per capability, so its Close must tolerate repeats.
func (c *Client) Close() error {
	var first error
	for _, sub := range []any{c.chatCompletion, c.embeddings, c.transcription, c.imageGeneration, c.decision, c.discovery} {
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

// RelayMessages implements [llm.MessagesRelayClient] when the chat completion
// client does.
func (c *Client) RelayMessages(ctx context.Context, body []byte, header http.Header) (<-chan llm.StreamChunk, error) {
	return llm.RelayMessages(ctx, c.chatCompletion, body, header)
}

// SupportsMessagesRelay reports on the chat completion client, see
// [llm.SupportsMessagesRelay].
func (c *Client) SupportsMessagesRelay() bool {
	return llm.SupportsMessagesRelay(c.chatCompletion)
}

var (
	_ llm.Client                = &Client{}
	_ llm.MessagesRelayClient   = &Client{}
	_ llm.ImageGenerationClient = &Client{}
	_ llm.DecisionClient        = &Client{}
	_ llm.DiscoveryClient       = &Client{}
	_ io.Closer                 = &Client{}
)

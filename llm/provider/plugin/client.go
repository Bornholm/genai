package plugin

import (
	"context"
	"io"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/plugin/codec"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/pkg/errors"
)

// ChatCompletionClient talks to a configured chat completion client inside a
// plugin process.
type ChatCompletionClient struct {
	proc     *Process
	clientID string
}

var _ llm.ChatCompletionClient = &ChatCompletionClient{}
var _ llm.ChatCompletionStreamingClient = &ChatCompletionClient{}

// NewChatCompletionClient starts (or reuses) the plugin at path and configures
// a chat completion client with the given options.
func NewChatCompletionClient(ctx context.Context, path string, options map[string]string) (*ChatCompletionClient, error) {
	proc, err := acquire(ctx, path)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	if !proc.supports(pluginv1.Capability_CAPABILITY_CHAT_COMPLETION) {
		return nil, errors.Errorf("plugin %q does not support chat completion", proc.info.GetName())
	}
	res, err := proc.clients.Provider.Configure(ctx, &pluginv1.ConfigureRequest{
		Capability: pluginv1.Capability_CAPABILITY_CHAT_COMPLETION,
		Options:    options,
	})
	if err != nil {
		return nil, errors.Wrapf(codec.ErrorFromStatus(err), "could not configure chat completion client of plugin %q", proc.info.GetName())
	}
	return &ChatCompletionClient{proc: proc, clientID: res.GetClientId()}, nil
}

// Process returns the underlying plugin process.
func (c *ChatCompletionClient) Process() *Process { return c.proc }

func (c *ChatCompletionClient) request(funcs []llm.ChatCompletionOptionFunc) (*pluginv1.ChatCompletionRequest, error) {
	opts := llm.NewChatCompletionOptions(funcs...)
	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}
	encoded, err := codec.ChatCompletionOptionsToProto(opts)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	return &pluginv1.ChatCompletionRequest{ClientId: c.clientID, Options: encoded}, nil
}

// ChatCompletion implements llm.ChatCompletionClient.
func (c *ChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	req, err := c.request(funcs)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	res, err := c.proc.clients.ChatCompletion.ChatCompletion(ctx, req)
	if err != nil {
		return nil, codec.ErrorFromStatus(err)
	}
	decoded, err := codec.ChatCompletionResponseFromProto(res)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	return decoded, nil
}

// ChatCompletionStream implements llm.ChatCompletionStreamingClient.
//
// Cancelling ctx cancels the gRPC stream, which cancels the server context on
// the plugin side and, through it, the provider's own upstream call.
func (c *ChatCompletionClient) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	req, err := c.request(funcs)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	stream, err := c.proc.clients.ChatCompletion.ChatCompletionStream(ctx, req)
	if err != nil {
		return nil, codec.ErrorFromStatus(err)
	}

	chunks := make(chan llm.StreamChunk, 10)

	go func() {
		defer close(chunks)

		terminated := false
		for {
			msg, err := stream.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) {
					if !terminated {
						llm.SendTerminalChunk(ctx, chunks, llm.NewErrorStreamChunk(errors.New("plugin closed the stream without a terminal chunk")))
					}
					return
				}
				if !terminated {
					llm.SendTerminalChunk(ctx, chunks, llm.NewErrorStreamChunk(codec.ErrorFromStatus(err)))
				}
				return
			}

			chunk, err := codec.StreamChunkFromProto(msg)
			if err != nil {
				llm.SendTerminalChunk(ctx, chunks, llm.NewErrorStreamChunk(errors.WithStack(err)))
				return
			}

			if chunk.IsComplete() || chunk.Type() == llm.StreamChunkTypeError {
				terminated = true
				llm.SendTerminalChunk(ctx, chunks, chunk)
				return
			}

			if !llm.SendChunk(ctx, chunks, chunk) {
				return
			}
		}
	}()

	return chunks, nil
}

// EmbeddingsClient talks to a configured embeddings client inside a plugin
// process.
type EmbeddingsClient struct {
	proc     *Process
	clientID string
}

var _ llm.EmbeddingsClient = &EmbeddingsClient{}

// NewEmbeddingsClient starts (or reuses) the plugin at path and configures an
// embeddings client with the given options.
func NewEmbeddingsClient(ctx context.Context, path string, options map[string]string) (*EmbeddingsClient, error) {
	proc, err := acquire(ctx, path)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	if !proc.supports(pluginv1.Capability_CAPABILITY_EMBEDDINGS) {
		return nil, errors.Errorf("plugin %q does not support embeddings", proc.info.GetName())
	}
	res, err := proc.clients.Provider.Configure(ctx, &pluginv1.ConfigureRequest{
		Capability: pluginv1.Capability_CAPABILITY_EMBEDDINGS,
		Options:    options,
	})
	if err != nil {
		return nil, errors.Wrapf(codec.ErrorFromStatus(err), "could not configure embeddings client of plugin %q", proc.info.GetName())
	}
	return &EmbeddingsClient{proc: proc, clientID: res.GetClientId()}, nil
}

// Process returns the underlying plugin process.
func (c *EmbeddingsClient) Process() *Process { return c.proc }

// Embeddings implements llm.EmbeddingsClient.
func (c *EmbeddingsClient) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	opts := llm.NewEmbeddingsOptions(funcs...)
	req := &pluginv1.EmbeddingsRequest{ClientId: c.clientID, Inputs: inputs}
	if opts.Dimensions != nil {
		dims := int64(*opts.Dimensions)
		req.Dimensions = &dims
	}
	res, err := c.proc.clients.Embeddings.Embeddings(ctx, req)
	if err != nil {
		return nil, codec.ErrorFromStatus(err)
	}
	return codec.EmbeddingsResponseFromProto(res), nil
}

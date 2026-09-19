package plugin

import (
	"context"
	"io"
	"sync"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/plugin/codec"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/pkg/errors"
)

// session is a configured client inside a plugin process. It remembers how it
// was configured so that it can configure itself again on a fresh process
// when the plugin died: a crash while loading a model is the likeliest
// failure of a native provider, and it should cost one failed call, not the
// rest of the host's life.
type session struct {
	path       string
	capability pluginv1.Capability
	options    map[string]string

	mu       sync.Mutex
	proc     *Process
	clientID string
	closed   bool
}

func newSession(ctx context.Context, path string, capability pluginv1.Capability, options map[string]string) (*session, error) {
	s := &session{path: path, capability: capability, options: options}
	if _, _, err := s.ensure(ctx); err != nil {
		return nil, errors.WithStack(err)
	}
	return s, nil
}

// ensure returns a live process and client id, reconfiguring after a crash.
func (s *session) ensure(ctx context.Context) (*Process, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, "", errors.Wrap(llm.ErrUnavailable, "plugin client is closed")
	}
	if s.proc != nil && !s.proc.Exited() {
		return s.proc, s.clientID, nil
	}

	proc, err := acquire(ctx, s.path)
	if err != nil {
		return nil, "", errors.Wrap(llm.ErrUnavailable, err.Error())
	}
	if !proc.supports(s.capability) {
		return nil, "", errors.Errorf("plugin %q does not support %s", proc.info.GetName(), s.capability.String())
	}
	clientID, err := proc.configure(ctx, s.capability, s.options)
	if err != nil {
		return nil, "", errors.Wrapf(err, "could not configure %s client of plugin %q", s.capability.String(), proc.info.GetName())
	}
	s.proc, s.clientID = proc, clientID
	return proc, clientID, nil
}

// close releases the client on the plugin side.
func (s *session) close(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.proc == nil {
		return nil
	}
	return s.proc.release(ctx, s.clientID)
}

func (s *session) process() *Process {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.proc
}

// ChatCompletionClient talks to a configured chat completion client inside a
// plugin process.
type ChatCompletionClient struct {
	session *session
}

var _ llm.ChatCompletionClient = &ChatCompletionClient{}
var _ llm.ChatCompletionStreamingClient = &ChatCompletionClient{}

// NewChatCompletionClient starts (or reuses) the plugin at path and configures
// a chat completion client with the given options.
func NewChatCompletionClient(ctx context.Context, path string, options map[string]string) (*ChatCompletionClient, error) {
	s, err := newSession(ctx, path, pluginv1.Capability_CAPABILITY_CHAT_COMPLETION, options)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	return &ChatCompletionClient{session: s}, nil
}

// Process returns the plugin process currently serving the client.
func (c *ChatCompletionClient) Process() *Process { return c.session.process() }

// Close releases the client on the plugin side. The plugin process keeps
// running for other clients until CleanupClients.
func (c *ChatCompletionClient) Close() error {
	return c.session.close(context.Background())
}

func (c *ChatCompletionClient) request(clientID string, funcs []llm.ChatCompletionOptionFunc) (*pluginv1.ChatCompletionRequest, error) {
	opts := llm.NewChatCompletionOptions(funcs...)
	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}
	encoded, err := codec.ChatCompletionOptionsToProto(opts)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	return &pluginv1.ChatCompletionRequest{ClientId: clientID, Options: encoded}, nil
}

// ChatCompletion implements llm.ChatCompletionClient.
func (c *ChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	proc, clientID, err := c.session.ensure(ctx)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	req, err := c.request(clientID, funcs)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	res, err := proc.clients.ChatCompletion.ChatCompletion(ctx, req)
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
	proc, clientID, err := c.session.ensure(ctx)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	req, err := c.request(clientID, funcs)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	// The gRPC stream is only released once its context is cancelled or Recv
	// returned an error. A terminal chunk does neither, so the stream gets a
	// context of its own, cancelled when the reader is done.
	streamCtx, cancel := context.WithCancel(ctx)
	stream, err := proc.clients.ChatCompletion.ChatCompletionStream(streamCtx, req)
	if err != nil {
		cancel()
		return nil, codec.ErrorFromStatus(err)
	}

	chunks := make(chan llm.StreamChunk, 10)

	go func() {
		defer close(chunks)
		defer cancel()

		for {
			msg, err := stream.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) {
					err = errors.New("plugin closed the stream without a terminal chunk")
				} else {
					err = codec.ErrorFromStatus(err)
				}
				llm.SendTerminalChunk(ctx, chunks, llm.NewErrorStreamChunk(err))
				return
			}

			chunk, err := codec.StreamChunkFromProto(msg)
			if err != nil {
				llm.SendTerminalChunk(ctx, chunks, llm.NewErrorStreamChunk(errors.WithStack(err)))
				return
			}

			if chunk.IsComplete() || chunk.Type() == llm.StreamChunkTypeError {
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
	session *session
}

var _ llm.EmbeddingsClient = &EmbeddingsClient{}

// NewEmbeddingsClient starts (or reuses) the plugin at path and configures an
// embeddings client with the given options.
func NewEmbeddingsClient(ctx context.Context, path string, options map[string]string) (*EmbeddingsClient, error) {
	s, err := newSession(ctx, path, pluginv1.Capability_CAPABILITY_EMBEDDINGS, options)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	return &EmbeddingsClient{session: s}, nil
}

// Process returns the plugin process currently serving the client.
func (c *EmbeddingsClient) Process() *Process { return c.session.process() }

// Close releases the client on the plugin side.
func (c *EmbeddingsClient) Close() error {
	return c.session.close(context.Background())
}

// Embeddings implements llm.EmbeddingsClient.
func (c *EmbeddingsClient) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	proc, clientID, err := c.session.ensure(ctx)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	opts := llm.NewEmbeddingsOptions(funcs...)
	req := &pluginv1.EmbeddingsRequest{ClientId: clientID, Inputs: inputs}
	if opts.Dimensions != nil {
		dims := int64(*opts.Dimensions)
		req.Dimensions = &dims
	}
	res, err := proc.clients.Embeddings.Embeddings(ctx, req)
	if err != nil {
		return nil, codec.ErrorFromStatus(err)
	}
	return codec.EmbeddingsResponseFromProto(res), nil
}

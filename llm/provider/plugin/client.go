package plugin

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/plugin/codec"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/pkg/errors"
)

// session is a configured client inside a plugin process. It remembers how it
// was configured so that it can configure itself again on a fresh process
// when the plugin died, or when a live plugin forgot the client: a crash
// while loading a model is the likeliest failure of a native provider, and
// it should cost one failed call, not the rest of the host's life.
type session struct {
	path       string
	capability pluginv1.Capability
	options    map[string]string

	// mu guards the fields below and is never held across an RPC, so that
	// Close and concurrent calls are not stuck behind a reconfiguration.
	mu       sync.Mutex
	proc     *Process
	clientID string
	closed   bool
	// generation counts reconfigurations; a caller that saw a failure on an
	// older generation does not trigger a second one.
	generation uint64

	// setup serializes reconfigurations.
	setup sync.Mutex
}

func newSession(ctx context.Context, path string, capability pluginv1.Capability, options map[string]string) (*session, error) {
	s := &session{path: path, capability: capability, options: options}
	if _, _, _, err := s.ensure(ctx); err != nil {
		return nil, errors.WithStack(err)
	}
	return s, nil
}

// current returns the live process and client id, if any.
func (s *session) current() (proc *Process, clientID string, generation uint64, closed bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.proc != nil && !s.proc.Exited() && s.clientID != "" {
		return s.proc, s.clientID, s.generation, s.closed
	}
	return nil, "", s.generation, s.closed
}

// ensure returns a live process and client id, reconfiguring after a crash
// or an invalidation.
func (s *session) ensure(ctx context.Context) (*Process, string, uint64, error) {
	proc, clientID, generation, closed := s.current()
	if closed {
		return nil, "", 0, errors.Wrap(llm.ErrUnavailable, "plugin client is closed")
	}
	if proc != nil {
		return proc, clientID, generation, nil
	}

	s.setup.Lock()
	defer s.setup.Unlock()

	// Another caller may have reconfigured while we waited.
	if proc, clientID, generation, closed := s.current(); closed || proc != nil {
		if closed {
			return nil, "", 0, errors.Wrap(llm.ErrUnavailable, "plugin client is closed")
		}
		return proc, clientID, generation, nil
	}

	proc, err := acquire(ctx, s.path)
	if err != nil {
		return nil, "", 0, errors.Wrap(llm.ErrUnavailable, err.Error())
	}
	if !proc.supports(s.capability) {
		return nil, "", 0, errors.Errorf("plugin %q does not support %s", proc.info.GetName(), s.capability.String())
	}
	clientID, err = proc.configure(ctx, s.capability, s.options)
	if err != nil {
		return nil, "", 0, errors.Wrapf(err, "could not configure %s client of plugin %q", s.capability.String(), proc.info.GetName())
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		// Closed while configuring, typically during a shutdown whose ctx
		// is already cancelled: release with a fresh one, release is bounded.
		_ = proc.release(context.Background(), clientID)
		return nil, "", 0, errors.Wrap(llm.ErrUnavailable, "plugin client is closed")
	}
	s.proc, s.clientID = proc, clientID
	s.generation++
	generation = s.generation
	s.mu.Unlock()
	return proc, clientID, generation, nil
}

// invalidate drops the client id of the given generation, so that the next
// call reconfigures. A newer generation is left alone.
func (s *session) invalidate(generation uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.generation == generation {
		s.clientID = ""
	}
}

// close releases the client on the plugin side.
func (s *session) close(ctx context.Context) error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	proc, clientID := s.proc, s.clientID
	s.mu.Unlock()
	if proc == nil || clientID == "" {
		return nil
	}
	return proc.release(ctx, clientID)
}

func (s *session) process() *Process {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.proc
}

// call runs fn against the live client, reconfiguring once when the plugin
// reports that it does not know the client.
func (s *session) call(ctx context.Context, fn func(proc *Process, clientID string) error) error {
	for attempt := 0; ; attempt++ {
		proc, clientID, generation, err := s.ensure(ctx)
		if err != nil {
			return errors.WithStack(err)
		}
		err = fn(proc, clientID)
		if err == nil || attempt > 0 || !errors.Is(err, codec.ErrUnknownClient) {
			return err
		}
		s.invalidate(generation)
	}
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
	var decoded llm.ChatCompletionResponse
	err := c.session.call(ctx, func(proc *Process, clientID string) error {
		req, err := c.request(clientID, funcs)
		if err != nil {
			return errors.WithStack(err)
		}
		res, err := proc.clients.ChatCompletion.ChatCompletion(ctx, req)
		if err != nil {
			return codec.ErrorFromStatus(err)
		}
		decoded, err = codec.ChatCompletionResponseFromProto(res)
		return errors.WithStack(err)
	})
	if err != nil {
		return nil, err
	}
	return decoded, nil
}

// ChatCompletionStream implements llm.ChatCompletionStreamingClient.
//
// Cancelling ctx cancels the gRPC stream, which cancels the server context on
// the plugin side and, through it, the provider's own upstream call.
//
// Unlike in-process providers, the call returns only once the first chunk
// has arrived: a plugin that forgot the client reports it on that first
// chunk, and the session reconfigures on it. The wait is bounded by
// StartTimeout, after which the call fails.
func (c *ChatCompletionClient) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	// The gRPC stream is only released once its context is cancelled or Recv
	// returned an error. A terminal chunk does neither, so the stream gets a
	// context of its own, cancelled when the reader is done.
	streamCtx, cancel := context.WithCancel(ctx)

	var stream pluginv1.ChatCompletion_ChatCompletionStreamClient
	err := c.session.call(ctx, func(proc *Process, clientID string) error {
		req, err := c.request(clientID, funcs)
		if err != nil {
			return errors.WithStack(err)
		}
		opened, err := proc.clients.ChatCompletion.ChatCompletionStream(streamCtx, req)
		if err != nil {
			return codec.ErrorFromStatus(err)
		}
		first, err := recvFirst(opened)
		if err != nil {
			return err
		}
		stream = &prefetchedStream{ChatCompletion_ChatCompletionStreamClient: opened, first: first}
		return nil
	})
	if err != nil {
		cancel()
		return nil, err
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

// recvFirst waits for the first chunk of a freshly opened stream, at most
// StartTimeout. Recv itself cannot take a context, so the wait runs aside
// and the stream is left to its context on timeout; the caller cancels it.
func recvFirst(stream pluginv1.ChatCompletion_ChatCompletionStreamClient) (*pluginv1.StreamChunk, error) {
	type result struct {
		chunk *pluginv1.StreamChunk
		err   error
	}
	done := make(chan result, 1)
	go func() {
		chunk, err := stream.Recv()
		done <- result{chunk, err}
	}()
	select {
	case r := <-done:
		if r.err != nil {
			return nil, codec.ErrorFromStatus(r.err)
		}
		return r.chunk, nil
	case <-time.After(StartTimeout):
		return nil, errors.Wrap(llm.ErrUnavailable, "plugin did not send a first chunk in time")
	}
}

// prefetchedStream hands back the first message read while opening the
// stream, then reads from the underlying stream.
type prefetchedStream struct {
	pluginv1.ChatCompletion_ChatCompletionStreamClient
	first *pluginv1.StreamChunk
}

func (s *prefetchedStream) Recv() (*pluginv1.StreamChunk, error) {
	if s.first != nil {
		first := s.first
		s.first = nil
		return first, nil
	}
	return s.ChatCompletion_ChatCompletionStreamClient.Recv()
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
	opts := llm.NewEmbeddingsOptions(funcs...)
	var decoded *codec.EmbeddingsResponse
	err := c.session.call(ctx, func(proc *Process, clientID string) error {
		req := &pluginv1.EmbeddingsRequest{ClientId: clientID, Inputs: inputs}
		if opts.Dimensions != nil {
			dims := int64(*opts.Dimensions)
			req.Dimensions = &dims
		}
		res, err := proc.clients.Embeddings.Embeddings(ctx, req)
		if err != nil {
			return codec.ErrorFromStatus(err)
		}
		decoded = codec.EmbeddingsResponseFromProto(res)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return decoded, nil
}

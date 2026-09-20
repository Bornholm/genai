package sdk

import (
	"context"
	"io"
	"strconv"
	"sync"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/plugin/codec"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/pkg/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// drainTimeout bounds how long a stream abandoned by the host is drained so
// that the provider's goroutine reaches its cleanup.
const drainTimeout = 5 * time.Second

type server struct {
	pluginv1.UnimplementedProviderServer
	cfg Config

	// mu guards the maps only. Factories run outside of it: loading a model
	// for a second client must not freeze the requests of the first.
	mu         sync.RWMutex
	nextID     int
	chat       map[string]llm.ChatCompletionClient
	embeddings map[string]llm.EmbeddingsClient
}

func newServer(cfg Config) *server {
	return &server{
		cfg:        cfg,
		chat:       map[string]llm.ChatCompletionClient{},
		embeddings: map[string]llm.EmbeddingsClient{},
	}
}

// Describe implements pluginv1.ProviderServer.
func (s *server) Describe(ctx context.Context, req *pluginv1.DescribeRequest) (*pluginv1.DescribeResponse, error) {
	res := &pluginv1.DescribeResponse{Name: s.cfg.Name, Version: s.cfg.Version}
	if s.cfg.ChatCompletion != nil {
		res.Capabilities = append(res.Capabilities, pluginv1.Capability_CAPABILITY_CHAT_COMPLETION)
	}
	if s.cfg.Embeddings != nil {
		res.Capabilities = append(res.Capabilities, pluginv1.Capability_CAPABILITY_EMBEDDINGS)
	}
	return res, nil
}

// Configure implements pluginv1.ProviderServer.
func (s *server) Configure(ctx context.Context, req *pluginv1.ConfigureRequest) (*pluginv1.ConfigureResponse, error) {
	opts := Options(req.GetOptions())
	if opts == nil {
		opts = Options{}
	}

	// The RPC context dies when Configure returns; a factory that keeps its
	// context (background goroutine, HTTP client) must not inherit that.
	factoryCtx := context.WithoutCancel(ctx)

	switch req.GetCapability() {
	case pluginv1.Capability_CAPABILITY_CHAT_COMPLETION:
		if s.cfg.ChatCompletion == nil {
			return nil, status.Error(codes.Unimplemented, "plugin does not offer chat completion")
		}
		client, err := s.cfg.ChatCompletion(factoryCtx, opts)
		if err != nil {
			return nil, codec.ErrorToStatus(err)
		}
		if err := abandoned(ctx, client); err != nil {
			return nil, err
		}
		s.mu.Lock()
		id := s.allocateID()
		s.chat[id] = client
		s.mu.Unlock()
		return s.configured(ctx, id)

	case pluginv1.Capability_CAPABILITY_EMBEDDINGS:
		if s.cfg.Embeddings == nil {
			return nil, status.Error(codes.Unimplemented, "plugin does not offer embeddings")
		}
		client, err := s.cfg.Embeddings(factoryCtx, opts)
		if err != nil {
			return nil, codec.ErrorToStatus(err)
		}
		if err := abandoned(ctx, client); err != nil {
			return nil, err
		}
		s.mu.Lock()
		id := s.allocateID()
		s.embeddings[id] = client
		s.mu.Unlock()
		return s.configured(ctx, id)

	default:
		return nil, status.Errorf(codes.InvalidArgument, "unknown capability %q", req.GetCapability().String())
	}
}

// abandoned closes a freshly built client when the host gave up on the
// Configure call meanwhile: gRPC would drop the response, the host would
// never learn the id, and the client would stay registered for the life of
// the process. It returns the status to answer with in that case.
func abandoned(ctx context.Context, client any) error {
	if ctx.Err() == nil {
		return nil
	}
	if closer, ok := client.(io.Closer); ok {
		_ = closer.Close()
	}
	return codec.ErrorToStatus(ctx.Err())
}

// configured answers a Configure once the client is registered, unless the
// host gave up in the meantime: the id would then never reach it, so the
// client is released right away. A cancellation landing after this point
// and before gRPC writes the response still leaks the client; that window
// is the transport's.
func (s *server) configured(ctx context.Context, id string) (*pluginv1.ConfigureResponse, error) {
	if ctx.Err() != nil {
		_, _ = s.Release(context.Background(), &pluginv1.ReleaseRequest{ClientId: id})
		return nil, codec.ErrorToStatus(ctx.Err())
	}
	return &pluginv1.ConfigureResponse{ClientId: id}, nil
}

// allocateID must be called with mu held for writing.
func (s *server) allocateID() string {
	s.nextID++
	return strconv.Itoa(s.nextID)
}

// Release implements pluginv1.ProviderServer. A client implementing io.Closer
// is closed, which is how a provider holding a model frees it. The host owes
// the plugin that no call is in flight for the client: the host side only
// releases from Close, after its own callers are done. The client is
// forgotten before Close runs, so that two concurrent Release calls close it
// once, and stays forgotten if Close fails: the host never retries, and a
// client nobody can reach is worse than a Close error reported once.
func (s *server) Release(ctx context.Context, req *pluginv1.ReleaseRequest) (*pluginv1.ReleaseResponse, error) {
	id := req.GetClientId()

	s.mu.Lock()
	var client any
	if c, ok := s.chat[id]; ok {
		client = c
		delete(s.chat, id)
	} else if c, ok := s.embeddings[id]; ok {
		client = c
		delete(s.embeddings, id)
	}
	s.mu.Unlock()

	if client == nil {
		return nil, status.Errorf(codes.NotFound, "unknown client %q", id)
	}
	if closer, ok := client.(io.Closer); ok {
		if err := closer.Close(); err != nil {
			return nil, codec.ErrorToStatus(err)
		}
	}
	return &pluginv1.ReleaseResponse{}, nil
}

func (s *server) chatCompletionServer() pluginv1.ChatCompletionServer {
	if s.cfg.ChatCompletion == nil {
		return nil
	}
	return &chatCompletionServer{s: s}
}

func (s *server) embeddingsServer() pluginv1.EmbeddingsServer {
	if s.cfg.Embeddings == nil {
		return nil
	}
	return &embeddingsServer{s: s}
}

func (s *server) chatClient(id string) (llm.ChatCompletionClient, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	client, ok := s.chat[id]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "unknown chat completion client %q", id)
	}
	return client, nil
}

func (s *server) embeddingsClient(id string) (llm.EmbeddingsClient, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	client, ok := s.embeddings[id]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "unknown embeddings client %q", id)
	}
	return client, nil
}

type chatCompletionServer struct {
	pluginv1.UnimplementedChatCompletionServer
	s *server
}

// ChatCompletion implements pluginv1.ChatCompletionServer.
func (c *chatCompletionServer) ChatCompletion(ctx context.Context, req *pluginv1.ChatCompletionRequest) (*pluginv1.ChatCompletionResponse, error) {
	client, err := c.s.chatClient(req.GetClientId())
	if err != nil {
		return nil, err
	}
	funcs, err := codec.ChatCompletionOptionsFromProto(req.GetOptions())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	res, err := client.ChatCompletion(ctx, funcs...)
	if err != nil {
		return nil, codec.ErrorToStatus(err)
	}
	encoded, err := codec.ChatCompletionResponseToProto(res)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return encoded, nil
}

// ChatCompletionStream implements pluginv1.ChatCompletionServer.
func (c *chatCompletionServer) ChatCompletionStream(req *pluginv1.ChatCompletionRequest, stream pluginv1.ChatCompletion_ChatCompletionStreamServer) error {
	client, err := c.s.chatClient(req.GetClientId())
	if err != nil {
		return err
	}
	funcs, err := codec.ChatCompletionOptionsFromProto(req.GetOptions())
	if err != nil {
		return status.Error(codes.InvalidArgument, err.Error())
	}

	ctx, cancel := context.WithCancel(stream.Context())
	defer cancel()

	streaming, ok := client.(llm.ChatCompletionStreamingClient)
	if !ok {
		return c.streamFromCompletion(ctx, client, funcs, stream)
	}

	chunks, err := streaming.ChatCompletionStream(ctx, funcs...)
	if err != nil {
		return codec.ErrorToStatus(err)
	}

	for {
		select {
		case <-ctx.Done():
			// The host went away: release the provider's goroutine before
			// leaving, or it stays stuck on a channel nobody reads.
			llm.DrainStream(chunks, drainTimeout)
			return status.FromContextError(ctx.Err()).Err()

		case chunk, ok := <-chunks:
			if !ok {
				// The provider closed without a terminal chunk, which the
				// contract forbids; tell the host rather than let it guess.
				return stream.Send(codec.StreamChunkToProto(llm.NewErrorStreamChunk(errors.New("provider closed the stream without a terminal chunk"))))
			}
			if err := stream.Send(codec.StreamChunkToProto(chunk)); err != nil {
				cancel()
				llm.DrainStream(chunks, drainTimeout)
				return err
			}
			if chunk.IsComplete() || chunk.Type() == llm.StreamChunkTypeError {
				return nil
			}
		}
	}
}

// streamFromCompletion answers a streaming request for a client that only
// implements ChatCompletion, as a single delta followed by a complete chunk.
// A stream delta carries text, reasoning and tool calls only: attachments
// and cache control of the response message do not fit and are dropped,
// which is why a provider answering multimodally should implement
// ChatCompletionStream itself.
func (c *chatCompletionServer) streamFromCompletion(ctx context.Context, client llm.ChatCompletionClient, funcs []llm.ChatCompletionOptionFunc, stream pluginv1.ChatCompletion_ChatCompletionStreamServer) error {
	res, err := client.ChatCompletion(ctx, funcs...)
	if err != nil {
		return stream.Send(codec.StreamChunkToProto(llm.NewErrorStreamChunk(err)))
	}

	var toolCallDeltas []llm.ToolCallDelta
	for i, tc := range res.ToolCalls() {
		params, err := codec.ToolCallToProto(tc)
		if err != nil {
			return status.Error(codes.Internal, err.Error())
		}
		toolCallDeltas = append(toolCallDeltas, llm.NewToolCallDelta(i, tc.ID(), tc.Name(), params.GetParametersJson()))
	}

	var delta llm.StreamDelta
	if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok && (rr.Reasoning() != "" || len(rr.ReasoningDetails()) > 0) {
		delta = llm.NewReasoningStreamDelta(llm.RoleAssistant, res.Message().Content(), rr.Reasoning(), rr.ReasoningDetails(), toolCallDeltas...)
	} else {
		delta = llm.NewStreamDelta(llm.RoleAssistant, res.Message().Content(), toolCallDeltas...)
	}

	if err := stream.Send(codec.StreamChunkToProto(llm.NewStreamChunk(delta))); err != nil {
		return err
	}
	return stream.Send(codec.StreamChunkToProto(llm.NewCompleteStreamChunk(res.Usage())))
}

type embeddingsServer struct {
	pluginv1.UnimplementedEmbeddingsServer
	s *server
}

// Embeddings implements pluginv1.EmbeddingsServer.
func (e *embeddingsServer) Embeddings(ctx context.Context, req *pluginv1.EmbeddingsRequest) (*pluginv1.EmbeddingsResponse, error) {
	client, err := e.s.embeddingsClient(req.GetClientId())
	if err != nil {
		return nil, err
	}
	var funcs []llm.EmbeddingsOptionFunc
	if req.Dimensions != nil {
		funcs = append(funcs, llm.WithDimensions(int(req.GetDimensions())))
	}
	res, err := client.Embeddings(ctx, req.GetInputs(), funcs...)
	if err != nil {
		return nil, codec.ErrorToStatus(err)
	}
	return codec.EmbeddingsResponseToProto(res), nil
}

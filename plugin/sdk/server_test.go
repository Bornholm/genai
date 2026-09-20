package sdk

import (
	"context"
	"errors"
	"testing"

	"github.com/bornholm/genai/llm"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// fakeClient is a chat client that only completes, counts its closes and
// can fail to close.
type fakeClient struct {
	closes    int
	closeErr  error
	toolCalls []llm.ToolCall
}

func (c *fakeClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return llm.NewChatCompletionResponseWithReasoning(llm.NewMessage(llm.RoleAssistant, "done"), llm.NewChatCompletionUsage(1, 2, 3), "why", nil, c.toolCalls...), nil
}

func (c *fakeClient) Close() error {
	c.closes++
	return c.closeErr
}

func newTestServer(client *fakeClient) *server {
	return newServer(Config{
		Name: "fake",
		ChatCompletion: func(ctx context.Context, opts Options) (llm.ChatCompletionClient, error) {
			return client, nil
		},
	})
}

func configure(t *testing.T, s *server) string {
	t.Helper()
	res, err := s.Configure(context.Background(), &pluginv1.ConfigureRequest{Capability: pluginv1.Capability_CAPABILITY_CHAT_COMPLETION})
	if err != nil {
		t.Fatalf("configure: %v", err)
	}
	return res.GetClientId()
}

func TestConfigureWithAbandonedContextClosesClient(t *testing.T) {
	client := &fakeClient{}
	s := newTestServer(client)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := s.Configure(ctx, &pluginv1.ConfigureRequest{Capability: pluginv1.Capability_CAPABILITY_CHAT_COMPLETION})
	if status.Code(err) != codes.Canceled {
		t.Errorf("expected a Canceled status, got %v", err)
	}
	if client.closes != 1 {
		t.Errorf("expected the abandoned client to be closed once, got %d", client.closes)
	}
	if len(s.chat) != 0 {
		t.Error("abandoned client stayed registered")
	}
}

func TestReleaseForgetsEvenWhenCloseFails(t *testing.T) {
	client := &fakeClient{closeErr: errors.New("unload failed")}
	s := newTestServer(client)
	id := configure(t, s)

	_, err := s.Release(context.Background(), &pluginv1.ReleaseRequest{ClientId: id})
	if err == nil {
		t.Fatal("expected the close error to be reported")
	}
	if _, ok := s.chat[id]; ok {
		t.Error("client stayed registered after a failed close")
	}
	_, err = s.Release(context.Background(), &pluginv1.ReleaseRequest{ClientId: id})
	if status.Code(err) != codes.NotFound {
		t.Errorf("second release should report NotFound, got %v", err)
	}
	if client.closes != 1 {
		t.Errorf("expected exactly one close, got %d", client.closes)
	}
}

// fakeStream records what the handler sends and can fail on send.
type fakeStream struct {
	grpc.ServerStream
	ctx     context.Context
	sent    []*pluginv1.StreamChunk
	sendErr error
}

func (s *fakeStream) Context() context.Context    { return s.ctx }
func (s *fakeStream) SetHeader(metadata.MD) error { return nil }
func (s *fakeStream) Send(chunk *pluginv1.StreamChunk) error {
	s.sent = append(s.sent, chunk)
	return s.sendErr
}

func TestStreamFromCompletionOnlyClient(t *testing.T) {
	client := &fakeClient{toolCalls: []llm.ToolCall{llm.NewToolCall("c1", "tool", `{"a":1}`)}}
	s := newTestServer(client)
	id := configure(t, s)

	stream := &fakeStream{ctx: context.Background()}
	req := &pluginv1.ChatCompletionRequest{ClientId: id, Options: &pluginv1.ChatCompletionOptions{
		Messages: []*pluginv1.Message{{Role: pluginv1.Role_ROLE_USER, Content: "hi"}},
	}}
	if err := s.chatCompletionServer().ChatCompletionStream(req, stream); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(stream.sent) != 2 {
		t.Fatalf("expected one delta and one complete chunk, got %d", len(stream.sent))
	}
	delta := stream.sent[0].GetDelta()
	if delta.GetContent() != "done" || delta.GetReasoning() != "why" || len(delta.GetToolCalls()) != 1 || delta.GetToolCalls()[0].GetParametersDelta() != `{"a":1}` {
		t.Errorf("unexpected delta %v", delta)
	}
	if stream.sent[1].GetType() != pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_COMPLETE || stream.sent[1].GetUsage().GetTotalTokens() != 3 {
		t.Errorf("unexpected terminal chunk %v", stream.sent[1])
	}
}

func TestStreamSendFailureIsReported(t *testing.T) {
	client := &fakeClient{}
	s := newTestServer(client)
	id := configure(t, s)

	stream := &fakeStream{ctx: context.Background(), sendErr: errors.New("host gone")}
	req := &pluginv1.ChatCompletionRequest{ClientId: id, Options: &pluginv1.ChatCompletionOptions{
		Messages: []*pluginv1.Message{{Role: pluginv1.Role_ROLE_USER, Content: "hi"}},
	}}
	if err := s.chatCompletionServer().ChatCompletionStream(req, stream); err == nil {
		t.Error("expected the send error to be returned")
	}
}

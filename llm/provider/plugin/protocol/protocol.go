// Package protocol holds what the host and a plugin binary must agree on:
// the go-plugin handshake, the plugin name and the gRPC services carried
// over the connection. Both llm/provider/plugin (host) and plugin/sdk
// (plugin side) build on it.
package protocol

import (
	"context"

	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/hashicorp/go-plugin"
	"google.golang.org/grpc"
)

// PluginName is the key under which the provider plugin is served.
const PluginName = "provider"

// ProtocolVersion is bumped on every incompatible change to provider.proto.
// A mismatch between host and plugin surfaces as a handshake error rather
// than as opaque gRPC failures.
const ProtocolVersion = 1

// ProtoChecksum is the SHA-256 of provider.proto at the time ProtocolVersion
// was last reviewed. A test compares it to the file: editing the proto means
// updating this constant, and deciding whether the change is compatible.
const ProtoChecksum = "c8825aad4b7c47eb057057e4c09706f18921ca660022e4c51a5fe12915dc5e2a"

// MaxMessageSize bounds a single gRPC message in both directions. Attachments
// travel base64 encoded inside messages, so the default 4 MiB is too small.
const MaxMessageSize = 64 << 20

// Handshake is checked by go-plugin before any RPC. It is a UX guard against
// running an arbitrary binary as a plugin, not a security feature.
var Handshake = plugin.HandshakeConfig{
	ProtocolVersion:  ProtocolVersion,
	MagicCookieKey:   "GENAI_PROVIDER_PLUGIN",
	MagicCookieValue: "d0c8a9a3-6c7e-4c5c-9e64-3a0f0c1b2e5f",
}

// BinaryPrefix is what a plugin binary name starts with; the provider name
// follows (genai-provider-yzma).
const BinaryPrefix = "genai-provider-"

// Servers groups the gRPC services a plugin implements. Provider is required;
// the others are registered when non-nil.
type Servers struct {
	Provider       pluginv1.ProviderServer
	ChatCompletion pluginv1.ChatCompletionServer
	Embeddings     pluginv1.EmbeddingsServer
}

// Clients groups the gRPC clients the host obtains from a connection.
type Clients struct {
	Provider       pluginv1.ProviderClient
	ChatCompletion pluginv1.ChatCompletionClient
	Embeddings     pluginv1.EmbeddingsClient
}

// Plugin is the go-plugin descriptor. The plugin side sets Servers; the host
// side leaves it nil and only uses GRPCClient.
type Plugin struct {
	plugin.NetRPCUnsupportedPlugin
	Servers *Servers
}

var _ plugin.GRPCPlugin = &Plugin{}

// GRPCServer implements plugin.GRPCPlugin.
func (p *Plugin) GRPCServer(broker *plugin.GRPCBroker, s *grpc.Server) error {
	if p.Servers == nil {
		return nil
	}
	if p.Servers.Provider != nil {
		pluginv1.RegisterProviderServer(s, p.Servers.Provider)
	}
	if p.Servers.ChatCompletion != nil {
		pluginv1.RegisterChatCompletionServer(s, p.Servers.ChatCompletion)
	}
	if p.Servers.Embeddings != nil {
		pluginv1.RegisterEmbeddingsServer(s, p.Servers.Embeddings)
	}
	return nil
}

// GRPCClient implements plugin.GRPCPlugin.
func (p *Plugin) GRPCClient(ctx context.Context, broker *plugin.GRPCBroker, c *grpc.ClientConn) (any, error) {
	return &Clients{
		Provider:       pluginv1.NewProviderClient(c),
		ChatCompletion: pluginv1.NewChatCompletionClient(c),
		Embeddings:     pluginv1.NewEmbeddingsClient(c),
	}, nil
}

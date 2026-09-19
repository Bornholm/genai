// Package sdk is what a provider plugin binary builds on. A plugin wraps
// ordinary llm clients and hands their factories to Serve:
//
//	func main() {
//		sdk.Serve(sdk.Config{
//			Name:    "acme",
//			Version: version,
//			ChatCompletion: func(ctx context.Context, opts sdk.Options) (llm.ChatCompletionClient, error) {
//				var o Options // a struct with `env:"..."` tags, as in-tree providers use
//				if err := opts.Decode(&o); err != nil {
//					return nil, err
//				}
//				return acme.NewChatCompletionClient(o.BaseURL, o.APIKey, o.Model), nil
//			},
//		})
//	}
//
// The host finds the binary as genai-provider-<name> and forwards every
// environment variable under the provider prefix as Options.
//
// Chat completion options arrive as the host built them, with one exception:
// numbers in ExtraFields are float64, as after decoding JSON, whatever their
// Go type was on the host.
package sdk

import (
	"context"
	"os"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/plugin/protocol"
	"github.com/hashicorp/go-hclog"
	goplugin "github.com/hashicorp/go-plugin"
	"google.golang.org/grpc"
)

// ChatCompletionFactory builds a chat completion client from its options. The
// returned client may also implement llm.ChatCompletionStreamingClient; when
// it does not, streaming requests are answered from ChatCompletion.
type ChatCompletionFactory func(ctx context.Context, opts Options) (llm.ChatCompletionClient, error)

// EmbeddingsFactory builds an embeddings client from its options.
type EmbeddingsFactory func(ctx context.Context, opts Options) (llm.EmbeddingsClient, error)

// Config describes a plugin.
type Config struct {
	// Name is the provider name, matching the binary suffix.
	Name string
	// Version is reported to the host, for diagnostics.
	Version string
	// ChatCompletion is set when the plugin offers chat completion.
	ChatCompletion ChatCompletionFactory
	// Embeddings is set when the plugin offers embeddings.
	Embeddings EmbeddingsFactory
	// Logger receives the plugin's own logs; they are relayed to the host's
	// stderr. Defaults to an info level logger.
	Logger hclog.Logger
}

// Serve runs the plugin until the host disconnects. It never returns.
func Serve(cfg Config) {
	logger := cfg.Logger
	if logger == nil {
		logger = hclog.New(&hclog.LoggerOptions{
			Name:   cfg.Name,
			Output: os.Stderr,
			Level:  hclog.Info,
		})
	}

	server := newServer(cfg)

	goplugin.Serve(&goplugin.ServeConfig{
		HandshakeConfig: protocol.Handshake,
		Plugins: map[string]goplugin.Plugin{
			protocol.PluginName: &protocol.Plugin{
				Servers: &protocol.Servers{
					Provider:       server,
					ChatCompletion: server.chatCompletionServer(),
					Embeddings:     server.embeddingsServer(),
				},
			},
		},
		GRPCServer: func(opts []grpc.ServerOption) *grpc.Server {
			opts = append(opts,
				grpc.MaxRecvMsgSize(protocol.MaxMessageSize),
				grpc.MaxSendMsgSize(protocol.MaxMessageSize),
			)
			return grpc.NewServer(opts...)
		},
		Logger: logger,
	})
}

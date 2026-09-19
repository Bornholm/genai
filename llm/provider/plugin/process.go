package plugin

import (
	"context"
	"log/slog"
	"os"
	"os/exec"
	"sync"

	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/bornholm/genai/llm/provider/plugin/protocol"
	"github.com/hashicorp/go-hclog"
	goplugin "github.com/hashicorp/go-plugin"
	"github.com/pkg/errors"
	"google.golang.org/grpc"
)

// Process is a running plugin binary. One process serves every capability the
// binary offers, so a provider like yzma loads its model once even when it is
// configured for both chat and embeddings.
type Process struct {
	path    string
	client  *goplugin.Client
	clients *protocol.Clients
	info    *pluginv1.DescribeResponse
}

// Path returns the binary the process runs.
func (p *Process) Path() string { return p.path }

// Info returns what the plugin reported at startup.
func (p *Process) Info() *pluginv1.DescribeResponse { return p.info }

// Kill stops the process. Configured clients become unusable.
func (p *Process) Kill() {
	p.client.Kill()
}

var (
	poolMu sync.Mutex
	pool   = map[string]*Process{}
)

// LogLevel is the level at which plugin logs are relayed to stderr.
var LogLevel = hclog.Info

// acquire returns the running process for a binary, starting it on first use.
func acquire(ctx context.Context, path string) (*Process, error) {
	poolMu.Lock()
	defer poolMu.Unlock()

	if proc, ok := pool[path]; ok && !proc.client.Exited() {
		return proc, nil
	}

	proc, err := start(ctx, path)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	pool[path] = proc
	return proc, nil
}

func start(ctx context.Context, path string) (*Process, error) {
	logger := hclog.New(&hclog.LoggerOptions{
		Name:   "plugin",
		Output: os.Stderr,
		Level:  LogLevel,
	})

	client := goplugin.NewClient(&goplugin.ClientConfig{
		HandshakeConfig:  protocol.Handshake,
		Plugins:          map[string]goplugin.Plugin{protocol.PluginName: &protocol.Plugin{}},
		Cmd:              exec.Command(path),
		AllowedProtocols: []goplugin.Protocol{goplugin.ProtocolGRPC},
		Managed:          true,
		Logger:           logger,
		GRPCDialOptions: []grpc.DialOption{
			grpc.WithDefaultCallOptions(
				grpc.MaxCallRecvMsgSize(protocol.MaxMessageSize),
				grpc.MaxCallSendMsgSize(protocol.MaxMessageSize),
			),
		},
	})

	rpcClient, err := client.Client()
	if err != nil {
		client.Kill()
		return nil, errors.Wrapf(err, "could not start plugin %q", path)
	}

	raw, err := rpcClient.Dispense(protocol.PluginName)
	if err != nil {
		client.Kill()
		return nil, errors.Wrapf(err, "could not dispense plugin %q", path)
	}

	clients, ok := raw.(*protocol.Clients)
	if !ok {
		client.Kill()
		return nil, errors.Errorf("plugin %q returned an unexpected client type %T", path, raw)
	}

	info, err := clients.Provider.Describe(ctx, &pluginv1.DescribeRequest{})
	if err != nil {
		client.Kill()
		return nil, errors.Wrapf(err, "could not describe plugin %q", path)
	}

	slog.DebugContext(ctx, "plugin started", slog.String("path", path), slog.String("name", info.GetName()), slog.String("version", info.GetVersion()))

	return &Process{path: path, client: client, clients: clients, info: info}, nil
}

func (p *Process) supports(capability pluginv1.Capability) bool {
	for _, c := range p.info.GetCapabilities() {
		if c == capability {
			return true
		}
	}
	return false
}

// CleanupClients kills every plugin process started by this package. Call it
// when the host exits.
func CleanupClients() {
	goplugin.CleanupClients()
	poolMu.Lock()
	defer poolMu.Unlock()
	pool = map[string]*Process{}
}

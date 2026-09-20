package plugin

import (
	"context"
	"log/slog"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/bornholm/genai/llm/provider/plugin/codec"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/bornholm/genai/llm/provider/plugin/protocol"
	"github.com/hashicorp/go-hclog"
	goplugin "github.com/hashicorp/go-plugin"
	"github.com/pkg/errors"
	"google.golang.org/grpc"
)

// Process is a running plugin binary. One process serves every capability the
// binary offers; each capability is still configured on its own, so whether
// a provider like yzma loads its model once for chat and embeddings is up
// to the plugin's factories sharing an instance.
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

// Exited reports whether the process is gone. Clients configured on it
// reconfigure themselves on a fresh process at their next call.
func (p *Process) Exited() bool { return p.client.Exited() }

// Kill stops the process. Clients configured on it reconfigure themselves on
// a fresh process at their next call.
func (p *Process) Kill() {
	p.client.Kill()
}

// configure asks the plugin for a client of the given capability.
func (p *Process) configure(ctx context.Context, capability pluginv1.Capability, options map[string]string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, ConfigureTimeout)
	defer cancel()
	res, err := p.clients.Provider.Configure(ctx, &pluginv1.ConfigureRequest{
		Capability: capability,
		Options:    options,
	})
	if err != nil {
		return "", codec.ErrorFromStatus(err)
	}
	return res.GetClientId(), nil
}

// release tells the plugin a client is no longer used.
func (p *Process) release(ctx context.Context, clientID string) error {
	if p.Exited() {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, StartTimeout)
	defer cancel()
	_, err := p.clients.Provider.Release(ctx, &pluginv1.ReleaseRequest{ClientId: clientID})
	if err != nil {
		return codec.ErrorFromStatus(err)
	}
	return nil
}

func (p *Process) supports(capability pluginv1.Capability) bool {
	for _, c := range p.info.GetCapabilities() {
		if c == capability {
			return true
		}
	}
	return false
}

// StartTimeout bounds the start of a plugin process (handshake and Describe)
// and the Release of a client: a binary that never answers fails instead of
// hanging the host.
var StartTimeout = 30 * time.Second

// ConfigureTimeout bounds each Configure call. It is separate from
// StartTimeout because configuring is where a native provider loads its
// model, which takes minutes for a large one.
var ConfigureTimeout = 5 * time.Minute

// FirstChunkTimeout bounds the wait for the first chunk of a stream, which
// is the provider's time to first token: for a local model with a long
// prompt that takes minutes, hence a budget aligned on ConfigureTimeout.
var FirstChunkTimeout = 5 * time.Minute

var (
	logLevelMu sync.RWMutex
	logLevel   = hclog.Info
)

// SetLogLevel sets the level at which plugin logs are relayed to stderr. It
// applies to plugins started afterwards.
func SetLogLevel(level hclog.Level) {
	logLevelMu.Lock()
	defer logLevelMu.Unlock()
	logLevel = level
}

// LogLevel returns the level at which plugin logs are relayed to stderr.
func LogLevel() hclog.Level {
	logLevelMu.RLock()
	defer logLevelMu.RUnlock()
	return logLevel
}

// pool holds one process per binary path. poolMu only guards the maps; the
// start of a given binary is serialized by its own lock so that a slow plugin
// does not block the start of another.
var (
	poolMu sync.Mutex
	pool   = map[string]*Process{}
	// locks holds one slot per binary path: a channel rather than a mutex so
	// that a caller waiting for another caller's start honours its context.
	locks = map[string]chan struct{}{}
)

func pathLock(path string) chan struct{} {
	poolMu.Lock()
	defer poolMu.Unlock()
	lock, ok := locks[path]
	if !ok {
		lock = make(chan struct{}, 1)
		locks[path] = lock
	}
	return lock
}

// Start returns the running process for a binary, starting it on first use
// or after it exited. Clients are normally obtained through
// NewChatCompletionClient and NewEmbeddingsClient; Start is for inspecting a
// plugin (Info) directly.
func Start(ctx context.Context, path string) (*Process, error) {
	return acquire(ctx, path)
}

func acquire(ctx context.Context, path string) (*Process, error) {
	lock := pathLock(path)
	select {
	case lock <- struct{}{}:
		defer func() { <-lock }()
	case <-ctx.Done():
		return nil, errors.WithStack(ctx.Err())
	}

	poolMu.Lock()
	proc, ok := pool[path]
	poolMu.Unlock()
	if ok && !proc.Exited() {
		return proc, nil
	}

	proc, err := start(ctx, path)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	poolMu.Lock()
	pool[path] = proc
	poolMu.Unlock()
	return proc, nil
}

func start(ctx context.Context, path string) (*Process, error) {
	logger := hclog.New(&hclog.LoggerOptions{
		Name:   "plugin",
		Output: os.Stderr,
		Level:  LogLevel(),
	})

	client := goplugin.NewClient(&goplugin.ClientConfig{
		HandshakeConfig:  protocol.Handshake,
		Plugins:          map[string]goplugin.Plugin{protocol.PluginName: &protocol.Plugin{}},
		Cmd:              exec.Command(path),
		AllowedProtocols: []goplugin.Protocol{goplugin.ProtocolGRPC},
		Managed:          true,
		Logger:           logger,
		StartTimeout:     StartTimeout,
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

	describeCtx, cancel := context.WithTimeout(ctx, StartTimeout)
	defer cancel()
	info, err := clients.Provider.Describe(describeCtx, &pluginv1.DescribeRequest{})
	if err != nil {
		client.Kill()
		return nil, errors.Wrapf(err, "could not describe plugin %q", path)
	}

	slog.DebugContext(ctx, "plugin started", slog.String("path", path), slog.String("name", info.GetName()), slog.String("version", info.GetVersion()))

	return &Process{path: path, client: client, clients: clients, info: info}, nil
}

// CleanupClients kills every plugin process started by this package. A host
// must call it before exiting: releasing clients does not stop their
// process, which stays warm for the next client of the same binary. Without
// it, plugin processes only die when go-plugin notices the host is gone,
// which is not a clean shutdown. It goes through go-plugin's own cleanup,
// which also kills plugin processes other libraries of the host started
// with go-plugin.
func CleanupClients() {
	goplugin.CleanupClients()
	poolMu.Lock()
	defer poolMu.Unlock()
	// locks is kept: an acquire in flight holds one of them, and a per-path
	// mutex is small and harmless.
	pool = map[string]*Process{}
}

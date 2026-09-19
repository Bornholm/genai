// Package plugin loads LLM providers served by external binaries over
// hashicorp/go-plugin (gRPC). Importing it registers a fallback in the
// provider registry: a provider name with no in-process registration is
// looked up as a genai-provider-<name> binary in the plugin directory
// (GENAI_PLUGIN_DIR or SetSearchDir) and then in the PATH.
//
// The fallback is inactive until a plugin directory is set. Setting it is
// the opt-in: from then on an unknown provider name runs a binary, which
// inherits the whole environment of the host (go-plugin passes it on, and a
// native provider needs PATH, HOME and its library paths) and receives the
// variables under the provider prefix as its options, API keys included. The
// directory and the PATH are trusted as much as the host binary itself; the
// handshake only guards against running the wrong kind of program.
//
// Discovery relies on POSIX execute bits; Windows is not a target.
//
// Every environment variable under the provider prefix is forwarded verbatim
// to the plugin, which validates them with its own option struct:
//
//	GENAI_CHAT_COMPLETION_PROVIDER=yzma
//	GENAI_CHAT_COMPLETION_YZMA_MODEL_PATH=/models/qwen.gguf
//
// COMMAND overrides the resolved binary path for one capability:
//
//	GENAI_CHAT_COMPLETION_YZMA_COMMAND=/opt/genai/genai-provider-yzma
package plugin

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/pkg/errors"
)

// CommandOption is the option key overriding the resolved binary path.
const CommandOption = "COMMAND"

// Options is what the registry hands to a plugin provider. Every variable
// under the provider prefix lands in Env; Command overrides the lookup.
type Options struct {
	Name    provider.Name
	Command string `env:"COMMAND"`
	Env     map[string]string
}

var _ provider.RawEnvConsumer = &Options{}

// SetRawEnv implements provider.RawEnvConsumer.
func (o *Options) SetRawEnv(vars map[string]string) {
	o.Env = vars
}

// Validate implements provider.Validator.
func (o *Options) Validate() error {
	if !IsValidName(o.Name) {
		return llm.NewValidationError("provider", "invalid plugin provider name")
	}
	return nil
}

func (o *Options) path() (string, error) {
	if o.Command != "" {
		return o.Command, nil
	}
	if command, ok := o.Env[CommandOption]; ok && command != "" {
		return command, nil
	}
	return Resolve(o.Name)
}

func (o *Options) forwarded() map[string]string {
	result := make(map[string]string, len(o.Env))
	for k, v := range o.Env {
		if k == CommandOption {
			continue
		}
		result[k] = v
	}
	return result
}

// NewOptions returns options for the named plugin provider, for programmatic
// configuration through provider.WithChatCompletion and friends.
func NewOptions(name provider.Name, env map[string]string) *Options {
	return &Options{Name: name, Env: env}
}

func init() {
	provider.RegisterFallback(fallback)
}

func fallback(capability provider.Capability, name provider.Name) (*provider.FallbackEntry, bool) {
	if SearchDir() == "" || !IsValidName(name) {
		return nil, false
	}

	newOptions := func() any {
		return &Options{Name: name}
	}

	switch capability {
	case provider.CapabilityChatCompletion:
		return &provider.FallbackEntry{
			NewOptions: newOptions,
			CreateClient: func(ctx context.Context, opts any) (any, error) {
				o := opts.(*Options)
				path, err := o.path()
				if err != nil {
					return nil, errors.WithStack(err)
				}
				return NewChatCompletionClient(ctx, path, o.forwarded())
			},
		}, true

	case provider.CapabilityEmbeddings:
		return &provider.FallbackEntry{
			NewOptions: newOptions,
			CreateClient: func(ctx context.Context, opts any) (any, error) {
				o := opts.(*Options)
				path, err := o.path()
				if err != nil {
					return nil, errors.WithStack(err)
				}
				return NewEmbeddingsClient(ctx, path, o.forwarded())
			},
		}, true

	default:
		return nil, false
	}
}

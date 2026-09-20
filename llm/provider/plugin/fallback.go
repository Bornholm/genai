// Package plugin loads LLM providers served by external binaries over
// hashicorp/go-plugin (gRPC). Importing it registers a fallback in the
// provider registry: a provider name with no in-process registration is
// looked up as a genai-provider-<name> binary in the plugin directory
// (GENAI_PLUGIN_DIR or SetSearchDir). The PATH is not searched.
//
// The fallback is inactive until a plugin directory is set. Setting it is
// the opt-in: from then on an unknown provider name runs a binary from that
// directory, which inherits the whole environment of the host (go-plugin
// passes it on, and a native provider needs PATH, HOME and its library
// paths) and receives the variables under the provider prefix as its
// options, API keys included. The directory is trusted as much as the host
// binary itself; the handshake only guards against running the wrong kind
// of program.
//
// Discovery relies on POSIX execute bits; Windows is not a target.
//
// Two behaviours differ from in-process providers. ChatCompletionStream
// returns only once the first chunk arrived (see FirstChunkTimeout). And a
// plugin that fails to start costs every call up to StartTimeout: nothing
// remembers a failed start, the caller's retry policy decides.
//
// Every environment variable under the provider prefix is forwarded verbatim
// to the plugin, which validates them with its own option struct:
//
//	GENAI_CHAT_COMPLETION_PROVIDER=yzma
//	GENAI_CHAT_COMPLETION_YZMA_MODEL_PATH=/models/qwen.gguf
//
// COMMAND overrides the resolved binary path for one capability. From the
// environment it needs the directory to be set as well; given in code
// (Options.Command) it enables the fallback on its own:
//
//	GENAI_CHAT_COMPLETION_YZMA_COMMAND=/opt/genai/genai-provider-yzma
package plugin

import (
	"context"
	"fmt"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/pkg/errors"
)

// CommandOption is the option key overriding the resolved binary path.
const CommandOption = "COMMAND"

// Options is what the registry hands to a plugin provider. Every variable
// under the provider prefix lands in Env; Command overrides the lookup.
type Options struct {
	// Name is the provider name; the registry fills it in, and options built
	// in code may leave it empty.
	Name    provider.Name
	Command string `env:"COMMAND"`
	Env     map[string]string

	// fromEnv records that the options were populated from the environment.
	fromEnv bool
}

var _ provider.RawEnvConsumer = &Options{}

// SetRawEnv implements provider.RawEnvConsumer.
func (o *Options) SetRawEnv(vars map[string]string) {
	o.Env = vars
	o.fromEnv = true
}

// Validate implements provider.Validator. The name is checked against the
// provider's when the client is created; here an empty one is fine.
func (o *Options) Validate() error {
	if o.Name != "" && !IsValidName(o.Name) {
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
// configuration. The fallback still needs its opt-in: a plugin directory
// (SetSearchDir or GENAI_PLUGIN_DIR), or a COMMAND entry naming the binary.
//
//	plugin.SetSearchDir("/opt/genai/plugins")
//	provider.WithChatCompletion("acme", plugin.NewOptions("acme", map[string]string{"API_KEY": key}))
func NewOptions(name provider.Name, env map[string]string) Options {
	return Options{Name: name, Env: env}
}

func init() {
	provider.RegisterFallback(fallback)
}

// enabled reports whether the fallback answers at all: once a plugin
// directory is set, or when the options were built in code with a COMMAND.
// A COMMAND arriving through the environment does not count on its own: a
// .env file can set it, and setting the directory is what turns
// configuration sources into a way to run binaries.
func enabled(opts *Options) bool {
	if SearchDir() != "" {
		return true
	}
	return opts != nil && opts.Command != "" && !opts.fromEnv
}

// pluginOptions checks the options the registry hands over. The fallback
// answers for any unknown name, so a caller may well have configured that
// name with another provider's option type.
func pluginOptions(name provider.Name, opts any) (*Options, error) {
	var o *Options
	switch v := opts.(type) {
	case *Options:
		o = v
	case **Options:
		// provider.WithChatCompletion takes the address of what it is given.
		o = *v
	}
	if o == nil {
		return nil, llm.NewValidationError("provider", fmt.Sprintf("provider %q has no in-process registration (missing import?) and, as a plugin, takes plugin.Options, not %T", name, opts))
	}
	if o.Name != "" && o.Name != name {
		return nil, llm.NewValidationError("provider", fmt.Sprintf("options for plugin %q were given to provider %q", o.Name, name))
	}
	if o.Name == "" {
		o.Name = name
	}
	if !enabled(o) {
		return nil, errors.Wrapf(ErrPluginNotFound, "provider %q has no in-process registration and no plugin directory is set", name)
	}
	return o, nil
}

func fallback(capability provider.Capability, name provider.Name) (*provider.FallbackEntry, bool) {
	if !IsValidName(name) {
		return nil, false
	}
	// The fallback answers for every valid name: whether it is enabled (a
	// plugin directory, or a COMMAND in the options) is only known once the
	// options are, in pluginOptions. An unknown name without either still
	// fails with ErrClientNotFound, which ErrPluginNotFound wraps.

	newOptions := func() any {
		return &Options{Name: name}
	}

	switch capability {
	case provider.CapabilityChatCompletion:
		return &provider.FallbackEntry{
			NewOptions: newOptions,
			CreateClient: func(ctx context.Context, opts any) (any, error) {
				o, err := pluginOptions(name, opts)
				if err != nil {
					return nil, errors.WithStack(err)
				}
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
				o, err := pluginOptions(name, opts)
				if err != nil {
					return nil, errors.WithStack(err)
				}
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

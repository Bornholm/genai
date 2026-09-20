package sdk

import (
	"github.com/caarlos0/env/v11"
	"github.com/pkg/errors"
)

// Options carries the provider options the host forwarded, keyed without
// their environment prefix: GENAI_CHAT_COMPLETION_ACME_API_KEY arrives as
// "API_KEY".
type Options map[string]string

// Get returns an option, empty when absent.
func (o Options) Get(key string) string {
	return o[key]
}

// Validator is implemented by option structs that check themselves; Decode
// calls it after populating the struct.
type Validator interface {
	Validate() error
}

// Decode populates target, a pointer to a struct with `env:"..."` tags, from
// the options, then validates it when it implements Validator. It lets a
// plugin reuse the option structs in-tree providers are written with.
func (o Options) Decode(target any) error {
	if err := env.ParseWithOptions(target, env.Options{Environment: o}); err != nil {
		return errors.WithStack(err)
	}
	if v, ok := target.(Validator); ok {
		if err := v.Validate(); err != nil {
			return errors.WithStack(err)
		}
	}
	return nil
}

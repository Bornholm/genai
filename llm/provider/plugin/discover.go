package plugin

import (
	"os"
	"path/filepath"
	"regexp"
	"sync"

	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/plugin/protocol"
	"github.com/pkg/errors"
)

// ErrPluginNotFound is returned when no binary serves the requested provider.
// It wraps provider.ErrClientNotFound, so callers checking for an unknown
// provider keep working once the plugin fallback is active.
var ErrPluginNotFound = errors.Wrap(provider.ErrClientNotFound, "plugin not found")

// SearchDirEnv names the environment variable holding the plugin directory.
const SearchDirEnv = "GENAI_PLUGIN_DIR"

var (
	searchDirMu sync.RWMutex
	searchDir   = os.Getenv(SearchDirEnv)
)

// SetSearchDir sets the directory searched for plugin binaries, and enables
// the registry fallback: with no directory set, unknown provider names stay
// unknown and no binary is run. It must be called before the provider
// options are resolved, since resolution is what triggers the lookup.
func SetSearchDir(dir string) {
	searchDirMu.Lock()
	defer searchDirMu.Unlock()
	searchDir = dir
}

// SearchDir returns the directory searched for plugin binaries, empty when
// unset.
func SearchDir() string {
	searchDirMu.RLock()
	defer searchDirMu.RUnlock()
	return searchDir
}

var validName = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

// IsValidName reports whether name can be a plugin provider name.
func IsValidName(name provider.Name) bool {
	return validName.MatchString(string(name))
}

// Resolve finds the binary serving the named provider in the search
// directory, and only there: the directory is the opt-in, and a binary
// elsewhere is named explicitly through COMMAND.
func Resolve(name provider.Name) (string, error) {
	if !IsValidName(name) {
		return "", errors.Wrapf(ErrPluginNotFound, "invalid provider name %q", name)
	}
	dir := SearchDir()
	if dir == "" {
		return "", errors.Wrapf(ErrPluginNotFound, "no plugin directory set ($%s) for provider %q", SearchDirEnv, name)
	}
	binary := protocol.BinaryPrefix + string(name)
	candidate := filepath.Join(dir, binary)
	if !isExecutable(candidate) {
		return "", errors.Wrapf(ErrPluginNotFound, "no executable %q in %s", binary, dir)
	}
	return candidate, nil
}

func isExecutable(path string) bool {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return false
	}
	return info.Mode()&0o111 != 0
}

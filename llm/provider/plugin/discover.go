package plugin

import (
	"os"
	"os/exec"
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

// SetSearchDir sets the directory searched before the PATH, and enables the
// registry fallback: with no directory set, unknown provider names stay
// unknown and no binary is run. It must be called before the provider
// options are resolved, since resolution is what triggers the lookup.
func SetSearchDir(dir string) {
	searchDirMu.Lock()
	defer searchDirMu.Unlock()
	searchDir = dir
}

// SearchDir returns the directory searched before the PATH, empty when unset.
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

// Resolve finds the binary serving the named provider: first in the search
// directory, then in the PATH.
func Resolve(name provider.Name) (string, error) {
	if !IsValidName(name) {
		return "", errors.Wrapf(ErrPluginNotFound, "invalid provider name %q", name)
	}
	binary := protocol.BinaryPrefix + string(name)

	if dir := SearchDir(); dir != "" {
		candidate := filepath.Join(dir, binary)
		if isExecutable(candidate) {
			return candidate, nil
		}
	}

	path, err := exec.LookPath(binary)
	if err != nil {
		return "", errors.Wrapf(ErrPluginNotFound, "no executable %q in %s or in PATH", binary, describeSearchDir())
	}
	return path, nil
}

func describeSearchDir() string {
	if dir := SearchDir(); dir != "" {
		return dir
	}
	return "$" + SearchDirEnv + " (unset)"
}

func isExecutable(path string) bool {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return false
	}
	return info.Mode()&0o111 != 0
}

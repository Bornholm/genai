package conformance

import (
	"os"
	"strings"
)

// HeadersFromEnv reads the headers a conformance run adds to its requests
// from the environment variable name, as "Name:value,Other-Name:value", the
// format of provider.CommonOptions.Headers. A gateway may require some, such
// as x-opencode-session on OpenCode Go.
func HeadersFromEnv(name string) map[string]string {
	raw := os.Getenv(name)
	if raw == "" {
		return nil
	}
	headers := map[string]string{}
	for _, pair := range strings.Split(raw, ",") {
		key, value, ok := strings.Cut(pair, ":")
		if ok && strings.TrimSpace(key) != "" {
			headers[strings.TrimSpace(key)] = strings.TrimSpace(value)
		}
	}
	return headers
}

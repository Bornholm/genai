// Package plugin adds dynamically loaded providers to the registry, on top
// of the in-process providers of the parent package. It is a separate import
// so that applications choose whether to embed go-plugin and gRPC:
//
//	import (
//		_ "github.com/bornholm/genai/llm/provider/all"
//		_ "github.com/bornholm/genai/llm/provider/all/plugin"
//	)
//
// Even once imported, the plugin fallback stays inactive until a plugin
// directory is set (GENAI_PLUGIN_DIR or plugin.SetSearchDir).
package plugin

import (
	_ "github.com/bornholm/genai/llm/provider/all"
	_ "github.com/bornholm/genai/llm/provider/plugin"
)

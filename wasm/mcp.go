//go:build js && wasm

package main

import (
	"context"
	"sync"
	"syscall/js"

	"github.com/bornholm/genai/llm"
	mcphttp "github.com/bornholm/genai/mcp/http"
)

type mcpToolSet struct {
	client *mcphttp.Client
	tools  []llm.Tool
}

var (
	toolsMu sync.Mutex
	toolSets = map[int]*mcpToolSet{}
	toolsID  = 0
)

func registerToolSet(ts *mcpToolSet) int {
	toolsMu.Lock()
	defer toolsMu.Unlock()
	toolsID++
	toolSets[toolsID] = ts
	return toolsID
}

func getToolSet(id int) (*mcpToolSet, bool) {
	toolsMu.Lock()
	defer toolsMu.Unlock()
	ts, ok := toolSets[id]
	return ts, ok
}

// jsCreateMCPTools se connecte à un serveur MCP (SSE) et retourne une Promise
// qui se résout avec un handle représentant les outils disponibles.
//
// Usage JS :
//
//	const tools = await genai.createMCPTools("http://localhost:3000/sse")
//	console.log(tools.count) // nombre d'outils disponibles
func jsCreateMCPTools(this js.Value, args []js.Value) any {
	if len(args) < 1 {
		panic("genai.createMCPTools: argument endpoint manquant")
	}
	endpoint := args[0].String()

	return promisify(func() (js.Value, error) {
		ctx := context.Background()
		client := mcphttp.NewClient(endpoint)

		if err := client.Start(ctx); err != nil {
			return js.Undefined(), err
		}

		tools, err := client.GetTools(ctx)
		if err != nil {
			_ = client.Stop()
			return js.Undefined(), err
		}

		id := registerToolSet(&mcpToolSet{client: client, tools: tools})

		obj := js.Global().Get("Object").New()
		obj.Set("_id", id)
		obj.Set("count", len(tools))
		return obj, nil
	})
}

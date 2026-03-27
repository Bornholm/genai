//go:build js && wasm

package main

import (
	"syscall/js"

	_ "github.com/bornholm/genai/llm/provider/mistral"
	_ "github.com/bornholm/genai/llm/provider/openai"
	_ "github.com/bornholm/genai/llm/provider/openrouter"
)

func main() {
	genai := js.Global().Get("Object").New()
	genai.Set("createClient", js.FuncOf(jsCreateClient))
	genai.Set("createMCPTools", js.FuncOf(jsCreateMCPTools))
	genai.Set("createAgent", js.FuncOf(jsCreateAgent))
	genai.Set("registerTool", js.FuncOf(jsRegisterTool))
	js.Global().Set("genai", genai)

	// Maintien du runtime Go actif
	select {}
}

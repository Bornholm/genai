//go:build js && wasm

package main

import (
	"context"
	"sync"
	"syscall/js"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/mistral"
	"github.com/bornholm/genai/llm/provider/openai"
	"github.com/bornholm/genai/llm/provider/openrouter"
	"github.com/pkg/errors"
)

var (
	clientsMu sync.Mutex
	clients   = map[int]llm.Client{}
	clientsID = 0
)

func registerClient(c llm.Client) int {
	clientsMu.Lock()
	defer clientsMu.Unlock()
	clientsID++
	clients[clientsID] = c
	return clientsID
}

func getClient(id int) (llm.Client, bool) {
	clientsMu.Lock()
	defer clientsMu.Unlock()
	c, ok := clients[id]
	return c, ok
}

// jsCreateClient crée un client LLM à partir d'un objet de configuration JS.
//
// Usage JS :
//
//	const client = genai.createClient({
//	  provider: "openai",   // "openai" | "mistral" | "openrouter"
//	  model:    "gpt-4o",
//	  baseURL:  "https://api.openai.com/v1",  // optionnel
//	  apiKey:   "sk-...",
//	})
func jsCreateClient(this js.Value, args []js.Value) any {
	if len(args) < 1 {
		panic("genai.createClient: argument de configuration manquant")
	}
	config := args[0]

	providerName := config.Get("provider").String()
	model := config.Get("model").String()

	var baseURL, apiKey string
	if v := config.Get("baseURL"); v.Type() == js.TypeString {
		baseURL = v.String()
	}
	if v := config.Get("apiKey"); v.Type() == js.TypeString {
		apiKey = v.String()
	}

	client, err := newClient(context.Background(), providerName, model, baseURL, apiKey)
	if err != nil {
		panic("genai.createClient: " + err.Error())
	}

	id := registerClient(client)
	obj := js.Global().Get("Object").New()
	obj.Set("_id", id)
	return obj
}

func newClient(ctx context.Context, providerName, model, baseURL, apiKey string) (llm.Client, error) {
	common := provider.CommonOptions{
		Model:  model,
		APIKey: apiKey,
	}

	switch provider.Name(providerName) {
	case "openai":
		opts := openai.Options{CommonOptions: common}
		if baseURL != "" {
			opts.BaseURL = baseURL
		}
		return provider.Create(ctx, provider.WithChatCompletion("openai", opts))

	case "mistral":
		opts := mistral.Options{CommonOptions: common}
		if baseURL != "" {
			opts.BaseURL = baseURL
		}
		return provider.Create(ctx, provider.WithChatCompletion("mistral", opts))

	case "openrouter":
		opts := openrouter.Options{CommonOptions: common}
		if baseURL != "" {
			opts.BaseURL = baseURL
		}
		return provider.Create(ctx, provider.WithChatCompletion("openrouter", opts))

	default:
		return nil, errors.Errorf("provider inconnu : %s (supportés : openai, mistral, openrouter)", providerName)
	}
}

//go:build js && wasm

package main

import (
	"context"
	"encoding/json"
	"sync"
	"syscall/js"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

var (
	jsToolsMu sync.Mutex
	jsTools   []llm.Tool
)

// jsExecMu sérialise les appels vers les callbacks JS.
//
// En WASM le scheduler Go est coopératif et tourne sur un unique thread OS.
// La boucle agentique exécute les outils en goroutines parallèles (wg.Wait).
// Si plusieurs goroutines bloquent simultanément sur des Promises JS, le
// scheduler doit pouvoir yield vers la JS event loop pour que les callbacks
// .then/.catch s'exécutent. Un seul callback JS "en vol" à la fois garantit
// que le scheduler yield correctement et évite tout deadlock.
var jsExecMu sync.Mutex

// jsRegisterTool enregistre un outil JS dans le registre global.
// L'outil sera automatiquement inclus dans tout agent créé après cet appel.
//
// Usage JS :
//
//	genai.registerTool(
//	  "update_email",
//	  "Remplace le corps du courriel en cours de rédaction.",
//	  JSON.stringify({
//	    type: "object",
//	    properties: { content: { type: "string", description: "Corps du courriel" } },
//	    required: ["content"],
//	  }),
//	  async (args) => {
//	    await browser.compose.setComposeDetails(tabId, { plainTextBody: args.content })
//	    return "Courriel mis à jour avec succès."
//	  }
//	)
func jsRegisterTool(this js.Value, args []js.Value) any {
	if len(args) < 4 {
		panic("genai.registerTool: arguments (name, description, schemaJSON, callback) requis")
	}

	name := args[0].String()
	description := args[1].String()
	schemaJSON := args[2].String()
	callback := args[3]

	var schema map[string]any
	if err := json.Unmarshal([]byte(schemaJSON), &schema); err != nil {
		panic("genai.registerTool: schemaJSON invalide : " + err.Error())
	}

	jsToolsMu.Lock()
	jsTools = append(jsTools, &jsTool{
		name:        name,
		description: description,
		parameters:  schema,
		callback:    callback,
	})
	jsToolsMu.Unlock()

	return nil
}

func getRegisteredJSTools() []llm.Tool {
	jsToolsMu.Lock()
	defer jsToolsMu.Unlock()
	result := make([]llm.Tool, len(jsTools))
	copy(result, jsTools)
	return result
}

// jsTool est un llm.Tool dont l'exécution délègue à un callback JavaScript.
type jsTool struct {
	name        string
	description string
	parameters  map[string]any
	callback    js.Value
}

func (t *jsTool) Name() string               { return t.name }
func (t *jsTool) Description() string        { return t.description }
func (t *jsTool) Parameters() map[string]any { return t.parameters }

func (t *jsTool) Execute(ctx context.Context, params map[string]any) (llm.ToolResult, error) {
	// Sérialiser les appels JS : un seul callback "en vol" à la fois.
	// La goroutine courante bloque sur jsExecMu jusqu'à ce que le callback
	// précédent soit terminé, ce qui garantit que le scheduler Go peut
	// toujours yield vers la JS event loop entre deux appels.
	jsExecMu.Lock()
	defer jsExecMu.Unlock()

	argsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	argsJS := js.Global().Get("JSON").Call("parse", string(argsJSON))
	jsResult := t.callback.Invoke(argsJS)

	// Cas 1 : le callback retourne une Promise
	if isPromise(jsResult) {
		resultCh := make(chan string, 1)
		errCh := make(chan error, 1)

		// Déclarer d'abord pour pouvoir les référencer dans les closures.
		var thenFn, catchFn js.Func

		thenFn = js.FuncOf(func(_ js.Value, cbArgs []js.Value) any {
			defer thenFn.Release()
			defer catchFn.Release()
			result := ""
			if len(cbArgs) > 0 && cbArgs[0].Type() == js.TypeString {
				result = cbArgs[0].String()
			}
			resultCh <- result
			return nil
		})
		catchFn = js.FuncOf(func(_ js.Value, cbArgs []js.Value) any {
			defer thenFn.Release()
			defer catchFn.Release()
			msg := "erreur inconnue lors de l'exécution de l'outil"
			if len(cbArgs) > 0 {
				msg = cbArgs[0].Call("toString").String()
			}
			errCh <- errors.New(msg)
			return nil
		})

		jsResult.Call("then", thenFn).Call("catch", catchFn)

		select {
		case r := <-resultCh:
			return llm.NewToolResult(r), nil
		case e := <-errCh:
			return nil, e
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Cas 2 : valeur synchrone
	if jsResult.Type() == js.TypeString {
		return llm.NewToolResult(jsResult.String()), nil
	}
	return llm.NewToolResult(""), nil
}

func isPromise(v js.Value) bool {
	if v.Type() != js.TypeObject || v.IsNull() || v.IsUndefined() {
		return false
	}
	return v.Get("then").Type() == js.TypeFunction
}

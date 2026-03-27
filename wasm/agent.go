//go:build js && wasm

package main

import (
	"context"
	"encoding/json"
	"syscall/js"

	"github.com/bornholm/genai/agent"
	"github.com/bornholm/genai/agent/loop"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// jsCreateAgent crée un agent ReAct et retourne un objet JS avec une méthode run().
//
// Options disponibles :
//
//	genai.createAgent(client, {
//	  // Prompt système
//	  systemPrompt: "Tu es un assistant email.",
//
//	  // Outils MCP (handle retourné par createMCPTools)
//	  tools: mcpTools,
//
//	  // Limites d'itérations et de tokens
//	  maxIterations:       20,     // défaut : 100
//	  maxTokens:           80000,  // défaut : 80000
//	  maxToolResultTokens: 4096,   // défaut : 10000
//
//	  // Température d'inférence (injectée sur chaque appel)
//	  temperature: 0.7,
//
//	  // Ratio de compression du contexte (fenêtre glissante)
//	  compressionRatio: 0.0125,  // défaut : 1/80
//
//	  // Étape de planification forcée avant la boucle
//	  forcePlanningStep: false,
//
//	  // Raisonnement (modèles supportant les reasoning tokens)
//	  reasoning: {
//	    effort:    "high",  // "xhigh"|"high"|"medium"|"low"|"minimal"|"none"
//	    maxTokens: 5000,    // optionnel
//	    enabled:   true,    // optionnel
//	    exclude:   false,   // optionnel — exclure le raisonnement de la réponse
//	  },
//
//	  // Approbation humaine avant l'exécution de certains outils
//	  approvalRequired: ["dangerous_tool"],  // ou ["*"] pour tous les outils
//	  approvalFunc: async (toolName, argsJSON) => {
//	    return confirm(`Autoriser "${toolName}" ?\n${argsJSON}`)
//	  },
//
//	  // Estimateur de tokens personnalisé (optionnel)
//	  tokenEstimator: (text) => Math.ceil(text.length / 4),
//	})
func jsCreateAgent(this js.Value, args []js.Value) any {
	if len(args) < 2 {
		panic("genai.createAgent: arguments (client, config) requis")
	}

	clientObj := args[0]
	config := args[1]

	clientID := clientObj.Get("_id").Int()
	client, ok := getClient(clientID)
	if !ok {
		panic("genai.createAgent: client invalide")
	}

	// Envelopper le client pour injecter la température si fournie, puis désactiver le
	// streaming via withNoStreaming (wrapper le plus externe) afin que la boucle agentique
	// utilise toujours le chemin non-streaming (plus stable en WASM).
	var llmClient llm.ChatCompletionClient = client
	if v := config.Get("temperature"); v.Type() == js.TypeNumber {
		llmClient = &withTemperature{ChatCompletionClient: llmClient, temp: v.Float()}
	}
	llmClient = &withNoStreaming{ChatCompletionClient: llmClient}

	loopOpts := []loop.OptionFunc{
		loop.WithClient(llmClient),
	}

	if v := config.Get("systemPrompt"); v.Type() == js.TypeString {
		loopOpts = append(loopOpts, loop.WithSystemPrompt(v.String()))
	}
	if v := config.Get("maxIterations"); v.Type() == js.TypeNumber {
		loopOpts = append(loopOpts, loop.WithMaxIterations(v.Int()))
	}
	if v := config.Get("maxTokens"); v.Type() == js.TypeNumber {
		loopOpts = append(loopOpts, loop.WithMaxTokens(v.Int()))
	}
	if v := config.Get("maxToolResultTokens"); v.Type() == js.TypeNumber {
		loopOpts = append(loopOpts, loop.WithMaxToolResultTokens(v.Int()))
	}
	if v := config.Get("compressionRatio"); v.Type() == js.TypeNumber {
		loopOpts = append(loopOpts, loop.WithCompressionRatio(v.Float()))
	}
	if v := config.Get("forcePlanningStep"); v.Type() == js.TypeBoolean {
		loopOpts = append(loopOpts, loop.WithForcePlanningStep(v.Bool()))
	}

	// Raisonnement
	if v := config.Get("reasoning"); v.Type() == js.TypeObject && !v.IsNull() && !v.IsUndefined() {
		loopOpts = append(loopOpts, loop.WithReasoningOptions(jsToReasoningOptions(v)))
	}

	// Approbation humaine
	if v := config.Get("approvalRequired"); v.Type() == js.TypeObject && !v.IsNull() {
		var toolNames []string
		length := v.Length()
		for i := range length {
			toolNames = append(toolNames, v.Index(i).String())
		}
		if len(toolNames) > 0 {
			loopOpts = append(loopOpts, loop.WithApprovalRequiredTools(toolNames...))
		}
	}
	if v := config.Get("approvalFunc"); v.Type() == js.TypeFunction {
		loopOpts = append(loopOpts, loop.WithApprovalFunc(jsApprovalFunc(v)))
	}

	// Estimateur de tokens personnalisé
	if v := config.Get("tokenEstimator"); v.Type() == js.TypeFunction {
		loopOpts = append(loopOpts, loop.WithTokenEstimator(func(s string) int {
			result := v.Invoke(s)
			if result.Type() == js.TypeNumber {
				return result.Int()
			}
			return len(s) / 4
		}))
	}

	// Fusionner les outils MCP (si fournis) et les outils JS globaux
	var tools []llm.Tool
	if v := config.Get("tools"); v.Type() == js.TypeObject && !v.IsNull() && !v.IsUndefined() {
		if ts, ok := getToolSet(v.Get("_id").Int()); ok {
			tools = append(tools, ts.tools...)
		}
	}
	tools = append(tools, getRegisteredJSTools()...)
	if len(tools) > 0 {
		loopOpts = append(loopOpts, loop.WithTools(tools...))
	}

	handler, err := loop.NewHandler(loopOpts...)
	if err != nil {
		panic("genai.createAgent: " + err.Error())
	}

	agentObj := js.Global().Get("Object").New()
	agentObj.Set("run", js.FuncOf(makeRunFunc(handler)))
	return agentObj
}

// makeRunFunc retourne la fonction JS run(message, onEvent, attachments?) => { promise, cancel }.
//
//	const task = agent.run(
//	  "Analyse ce document",
//	  (type, data) => { if (type === "text_delta") process(data.delta) },
//	  [{ type: "image", mimeType: "image/png", source: "base64", data: "iVBOR..." }]
//	)
//	cancelBtn.onclick = () => task.cancel()
//	await task.promise
func makeRunFunc(handler *loop.Handler) func(js.Value, []js.Value) any {
	return func(this js.Value, args []js.Value) any {
		if len(args) < 1 {
			panic("agent.run: argument message manquant")
		}
		message := args[0].String()

		var onEvent js.Value
		if len(args) >= 2 && args[1].Type() == js.TypeFunction {
			onEvent = args[1]
		}

		var attachments []llm.Attachment
		if len(args) >= 3 && args[2].Type() == js.TypeObject && !args[2].IsNull() {
			attachments = jsToAttachments(args[2])
		}

		ctx, cancel := context.WithCancel(context.Background())

		promise := promisify(func() (js.Value, error) {
			runner := agent.NewRunner(handler)
			input := agent.NewInput(message, attachments...)

			emit := func(ev agent.Event) error {
				if !onEvent.IsUndefined() && !onEvent.IsNull() {
					data := marshalEventData(ev.Data())
					onEvent.Invoke(string(ev.Type()), data)
				}
				return nil
			}

			if err := runner.Run(ctx, input, emit); err != nil {
				if errors.Is(err, context.Canceled) {
					return js.Undefined(), nil
				}
				return js.Undefined(), err
			}
			return js.Undefined(), nil
		})

		cancelFn := js.FuncOf(func(_ js.Value, _ []js.Value) any {
			cancel()
			return nil
		})

		result := js.Global().Get("Object").New()
		result.Set("promise", promise)
		result.Set("cancel", cancelFn)
		return result
	}
}

// jsToReasoningOptions convertit un objet JS en *llm.ReasoningOptions.
func jsToReasoningOptions(v js.Value) *llm.ReasoningOptions {
	effort := llm.ReasoningEffortMedium
	if e := v.Get("effort"); e.Type() == js.TypeString {
		effort = llm.ReasoningEffort(e.String())
	}
	opts := llm.NewReasoningOptions(effort)
	if maxTok := v.Get("maxTokens"); maxTok.Type() == js.TypeNumber {
		n := maxTok.Int()
		opts.MaxTokens = &n
	}
	if enabled := v.Get("enabled"); enabled.Type() == js.TypeBoolean {
		b := enabled.Bool()
		opts.Enabled = &b
	}
	if exclude := v.Get("exclude"); exclude.Type() == js.TypeBoolean {
		opts.Exclude = exclude.Bool()
	}
	return opts
}

// jsApprovalFunc adapte un callback JS en loop.ApprovalFunc.
// Le callback JS reçoit (toolName: string, argsJSON: string) et doit retourner
// une Promise<boolean> ou un boolean.
func jsApprovalFunc(fn js.Value) loop.ApprovalFunc {
	return func(ctx interface{ Done() <-chan struct{} }, toolName string, arguments string) (bool, error) {
		jsResult := fn.Invoke(toolName, arguments)

		if isPromise(jsResult) {
			approvedCh := make(chan bool, 1)
			errCh := make(chan error, 1)

			var thenFn, catchFn js.Func
			thenFn = js.FuncOf(func(_ js.Value, cbArgs []js.Value) any {
				defer thenFn.Release()
				defer catchFn.Release()
				approved := len(cbArgs) > 0 && cbArgs[0].Truthy()
				approvedCh <- approved
				return nil
			})
			catchFn = js.FuncOf(func(_ js.Value, cbArgs []js.Value) any {
				defer thenFn.Release()
				defer catchFn.Release()
				msg := "erreur dans approvalFunc"
				if len(cbArgs) > 0 {
					msg = cbArgs[0].Call("toString").String()
				}
				errCh <- errors.New(msg)
				return nil
			})
			jsResult.Call("then", thenFn).Call("catch", catchFn)

			select {
			case approved := <-approvedCh:
				return approved, nil
			case err := <-errCh:
				return false, err
			case <-ctx.Done():
				return false, errors.New("contexte annulé")
			}
		}

		return jsResult.Truthy(), nil
	}
}

// jsToAttachments convertit un tableau JS d'attachements en []llm.Attachment.
//
// Format attendu pour chaque élément :
//
//	{ type: "image"|"audio"|"video"|"document", mimeType: string, source: "base64"|"url", data: string }
func jsToAttachments(arr js.Value) []llm.Attachment {
	length := arr.Length()
	result := make([]llm.Attachment, 0, length)
	for i := range length {
		item := arr.Index(i)
		attachType := item.Get("type").String()
		mimeType := item.Get("mimeType").String()
		source := item.Get("source").String()
		data := item.Get("data").String()
		isURL := source == "url"

		var att llm.Attachment
		var err error
		switch attachType {
		case "image":
			att, err = llm.NewImageAttachment(mimeType, data, isURL)
		case "audio":
			att, err = llm.NewAudioAttachment(mimeType, data, isURL)
		case "video":
			att, err = llm.NewVideoAttachment(mimeType, data, isURL)
		case "document":
			att, err = llm.NewDocumentAttachment(mimeType, data, isURL)
		}
		if err == nil && att != nil {
			result = append(result, att)
		}
	}
	return result
}

// marshalEventData sérialise la donnée d'un événement en objet JS.
func marshalEventData(data any) js.Value {
	if data == nil {
		return js.Null()
	}
	b, err := json.Marshal(data)
	if err != nil {
		return js.Null()
	}
	return js.Global().Get("JSON").Call("parse", string(b))
}

// withNoStreaming enveloppe un client LLM et masque l'interface
// ChatCompletionStreamingClient pour forcer le chemin non-streaming dans la boucle.
type withNoStreaming struct {
	llm.ChatCompletionClient
}

// withTemperature enveloppe un client LLM pour injecter une température fixe.
// Les deux interfaces (non-streaming et streaming) sont forwarded pour que la
// boucle agentique utilise le streaming quand le provider le supporte.
type withTemperature struct {
	llm.ChatCompletionClient
	temp float64
}

func (w *withTemperature) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return w.ChatCompletionClient.ChatCompletion(ctx, append(funcs, llm.WithTemperature(w.temp))...)
}

// ChatCompletionStream implémente llm.ChatCompletionStreamingClient en déléguant
// au client sous-jacent si celui-ci supporte le streaming.
func (w *withTemperature) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	sc, ok := w.ChatCompletionClient.(llm.ChatCompletionStreamingClient)
	if !ok {
		return nil, llm.ErrUnavailable
	}
	return sc.ChatCompletionStream(ctx, append(funcs, llm.WithTemperature(w.temp))...)
}

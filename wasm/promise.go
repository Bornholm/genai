//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"
)

// promisify exécute fn dans une goroutine et retourne une Promise JS.
// Les paniques Go sont capturées et converties en rejets de Promise.
func promisify(fn func() (js.Value, error)) js.Value {
	handler := js.FuncOf(func(this js.Value, args []js.Value) any {
		resolve := args[0]
		reject := args[1]
		go func() {
			defer func() {
				if r := recover(); r != nil {
					jsErr := js.Global().Get("Error").New(fmt.Sprintf("panic: %v", r))
					reject.Invoke(jsErr)
				}
			}()
			result, err := fn()
			if err != nil {
				jsErr := js.Global().Get("Error").New(err.Error())
				reject.Invoke(jsErr)
				return
			}
			resolve.Invoke(result)
		}()
		return nil
	})
	return js.Global().Get("Promise").New(handler)
}

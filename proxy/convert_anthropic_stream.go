package proxy

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/bornholm/genai/llm"
	"github.com/google/uuid"
	"github.com/pkg/errors"
)

// anthropicStreamEmitter encodes llm.StreamChunk values as Anthropic Messages
// SSE events (message_start, content_block_start/delta/stop, message_delta,
// message_stop).
type anthropicStreamEmitter struct {
	messageID string
	model     string

	// blockIndex is the index of the most recently opened content block, or
	// -1 if no block has been opened yet.
	blockIndex int
	// blockType is the type of the currently open block ("text", "thinking",
	// "tool_use"), or "" if the current block has been closed.
	blockType string
	// toolBlocks maps a ToolCallDelta.Index() to its Anthropic content block index.
	toolBlocks map[int]int

	sawToolCalls bool
}

func newAnthropicStreamEmitter(model string) *anthropicStreamEmitter {
	return &anthropicStreamEmitter{
		messageID:  "msg_" + uuid.New().String(),
		model:      model,
		blockIndex: -1,
		toolBlocks: make(map[int]int),
	}
}

// EmitFirst implements streamEmitter.
func (e *anthropicStreamEmitter) EmitFirst(w io.Writer, chunk llm.StreamChunk) error {
	var inputTokens int64
	if usage := chunk.Usage(); usage != nil {
		inputTokens = usage.PromptTokens()
	}

	if err := writeAnthropicSSEEvent(w, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            e.messageID,
			"type":          "message",
			"role":          "assistant",
			"model":         e.model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  inputTokens,
				"output_tokens": 0,
			},
		},
	}); err != nil {
		return err
	}

	return e.process(w, chunk)
}

// Emit implements streamEmitter.
func (e *anthropicStreamEmitter) Emit(w io.Writer, chunk llm.StreamChunk) error {
	return e.process(w, chunk)
}

// EmitError implements streamEmitter.
func (e *anthropicStreamEmitter) EmitError(w io.Writer, err error) error {
	apiErr := apiErrorFromErr(err)
	return writeAnthropicSSEEvent(w, "error", map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    anthropicErrorType(apiErr),
			"message": apiErr.Message,
		},
	})
}

// Finalize implements streamEmitter.
func (e *anthropicStreamEmitter) Finalize(w io.Writer, usage llm.ChatCompletionUsage) error {
	if err := e.closeBlock(w); err != nil {
		return err
	}

	stopReason := "end_turn"
	if e.sawToolCalls {
		stopReason = "tool_use"
	}

	var outputTokens int64
	if usage != nil {
		outputTokens = usage.CompletionTokens()
	}

	if err := writeAnthropicSSEEvent(w, "message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"output_tokens": outputTokens,
		},
	}); err != nil {
		return err
	}

	return writeAnthropicSSEEvent(w, "message_stop", map[string]any{
		"type": "message_stop",
	})
}

var _ streamEmitter = &anthropicStreamEmitter{}

// process emits content_block_* events for the delta carried by chunk, and
// closes the last open block once the stream is complete.
func (e *anthropicStreamEmitter) process(w io.Writer, chunk llm.StreamChunk) error {
	if delta := chunk.Delta(); delta != nil {
		if rd, ok := delta.(llm.ReasoningStreamDelta); ok {
			if reasoning := rd.Reasoning(); reasoning != "" {
				if err := e.ensureBlock(w, "thinking"); err != nil {
					return err
				}
				if err := writeAnthropicSSEEvent(w, "content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": e.blockIndex,
					"delta": map[string]any{"type": "thinking_delta", "thinking": reasoning},
				}); err != nil {
					return err
				}
			}

			for _, d := range rd.ReasoningDetails() {
				if d.Signature == "" {
					continue
				}
				if err := e.ensureBlock(w, "thinking"); err != nil {
					return err
				}
				if err := writeAnthropicSSEEvent(w, "content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": e.blockIndex,
					"delta": map[string]any{"type": "signature_delta", "signature": d.Signature},
				}); err != nil {
					return err
				}
			}
		}

		if content := delta.Content(); content != "" {
			if err := e.ensureBlock(w, "text"); err != nil {
				return err
			}
			if err := writeAnthropicSSEEvent(w, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": e.blockIndex,
				"delta": map[string]any{"type": "text_delta", "text": content},
			}); err != nil {
				return err
			}
		}

		for _, tc := range delta.ToolCalls() {
			e.sawToolCalls = true

			idx, err := e.ensureToolBlock(w, tc)
			if err != nil {
				return err
			}

			if partial := tc.ParametersDelta(); partial != "" {
				if err := writeAnthropicSSEEvent(w, "content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": idx,
					"delta": map[string]any{"type": "input_json_delta", "partial_json": partial},
				}); err != nil {
					return err
				}
			}
		}
	}

	if chunk.IsComplete() {
		if err := e.closeBlock(w); err != nil {
			return err
		}
	}

	return nil
}

// ensureBlock makes sure a content block of the given type ("text" or
// "thinking") is currently open, closing and opening blocks as needed.
func (e *anthropicStreamEmitter) ensureBlock(w io.Writer, blockType string) error {
	if e.blockType == blockType {
		return nil
	}

	if err := e.closeBlock(w); err != nil {
		return err
	}

	e.blockIndex++
	e.blockType = blockType

	var contentBlock map[string]any
	switch blockType {
	case "thinking":
		contentBlock = map[string]any{"type": "thinking", "thinking": ""}
	default:
		contentBlock = map[string]any{"type": "text", "text": ""}
	}

	return writeAnthropicSSEEvent(w, "content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         e.blockIndex,
		"content_block": contentBlock,
	})
}

// ensureToolBlock makes sure a tool_use content block exists for tc.Index(),
// opening a new one (closing the current block first) if needed. It returns
// the Anthropic content block index for this tool call.
func (e *anthropicStreamEmitter) ensureToolBlock(w io.Writer, tc llm.ToolCallDelta) (int, error) {
	if idx, ok := e.toolBlocks[tc.Index()]; ok && e.blockType == "tool_use" && e.blockIndex == idx {
		return idx, nil
	}

	if err := e.closeBlock(w); err != nil {
		return 0, err
	}

	e.blockIndex++
	e.blockType = "tool_use"
	e.toolBlocks[tc.Index()] = e.blockIndex

	if err := writeAnthropicSSEEvent(w, "content_block_start", map[string]any{
		"type":  "content_block_start",
		"index": e.blockIndex,
		"content_block": map[string]any{
			"type":  "tool_use",
			"id":    tc.ID(),
			"name":  tc.Name(),
			"input": map[string]any{},
		},
	}); err != nil {
		return 0, err
	}

	return e.blockIndex, nil
}

// closeBlock emits a content_block_stop event for the currently open block,
// if any. It is a no-op if no block is open.
func (e *anthropicStreamEmitter) closeBlock(w io.Writer) error {
	if e.blockType == "" {
		return nil
	}

	err := writeAnthropicSSEEvent(w, "content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": e.blockIndex,
	})
	e.blockType = ""
	return err
}

// writeAnthropicSSEEvent writes a single named SSE event with a JSON payload.
func writeAnthropicSSEEvent(w io.Writer, event string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return errors.WithStack(err)
	}

	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data); err != nil {
		return errors.WithStack(err)
	}

	return nil
}

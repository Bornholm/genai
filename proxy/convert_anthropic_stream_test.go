package proxy

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

// sseEvent is a parsed "event: ...\ndata: ...\n\n" block.
type sseEvent struct {
	Event string
	Data  map[string]any
}

// parseSSEEvents splits raw SSE output into individual events.
func parseSSEEvents(t *testing.T, raw string) []sseEvent {
	t.Helper()

	var events []sseEvent
	for _, block := range strings.Split(strings.TrimSpace(raw), "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}

		var event sseEvent
		for _, line := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(line, "event: "):
				event.Event = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				data := strings.TrimPrefix(line, "data: ")
				if err := json.Unmarshal([]byte(data), &event.Data); err != nil {
					t.Fatalf("could not unmarshal event data %q: %v", data, err)
				}
			}
		}
		events = append(events, event)
	}
	return events
}

func TestAnthropicStreamEmitter_TextOnly(t *testing.T) {
	emitter := newAnthropicStreamEmitter("claude-3-5-sonnet-20241022")
	var buf bytes.Buffer

	chunk1 := llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "Hello"))
	chunk2 := llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, " world"))
	complete := llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(10, 5, 15))

	if err := emitter.EmitFirst(&buf, chunk1); err != nil {
		t.Fatalf("EmitFirst: %v", err)
	}
	if err := emitter.Emit(&buf, chunk2); err != nil {
		t.Fatalf("Emit chunk2: %v", err)
	}
	if err := emitter.Emit(&buf, complete); err != nil {
		t.Fatalf("Emit complete: %v", err)
	}
	if err := emitter.Finalize(&buf, complete.Usage()); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	events := parseSSEEvents(t, buf.String())

	wantTypes := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}
	if len(events) != len(wantTypes) {
		t.Fatalf("events = %d, want %d: %+v", len(events), len(wantTypes), events)
	}
	for i, want := range wantTypes {
		if events[i].Event != want {
			t.Errorf("event[%d] = %q, want %q", i, events[i].Event, want)
		}
	}

	// content_block_start should describe a text block at index 0.
	contentBlock := events[1].Data["content_block"].(map[string]any)
	if contentBlock["type"] != "text" {
		t.Errorf("content_block.type = %v, want text", contentBlock["type"])
	}
	if events[1].Data["index"] != float64(0) {
		t.Errorf("content_block_start.index = %v, want 0", events[1].Data["index"])
	}

	// deltas carry the streamed text in order.
	delta1 := events[2].Data["delta"].(map[string]any)
	if delta1["type"] != "text_delta" || delta1["text"] != "Hello" {
		t.Errorf("delta1 = %+v", delta1)
	}
	delta2 := events[3].Data["delta"].(map[string]any)
	if delta2["type"] != "text_delta" || delta2["text"] != " world" {
		t.Errorf("delta2 = %+v", delta2)
	}

	// content_block_stop closes index 0.
	if events[4].Data["index"] != float64(0) {
		t.Errorf("content_block_stop.index = %v, want 0", events[4].Data["index"])
	}

	// message_delta carries the final stop reason and usage.
	msgDelta := events[5].Data["delta"].(map[string]any)
	if msgDelta["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v, want end_turn", msgDelta["stop_reason"])
	}
	msgUsage := events[5].Data["usage"].(map[string]any)
	if msgUsage["output_tokens"] != float64(5) {
		t.Errorf("output_tokens = %v, want 5", msgUsage["output_tokens"])
	}
}

func TestAnthropicStreamEmitter_ToolUse(t *testing.T) {
	emitter := newAnthropicStreamEmitter("claude-3-5-sonnet-20241022")
	var buf bytes.Buffer

	chunk1 := llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "",
		llm.NewToolCallDelta(0, "toolu_1", "get_weather", "")))
	chunk2 := llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "",
		llm.NewToolCallDelta(0, "", "", `{"location":`)))
	chunk3 := llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "",
		llm.NewToolCallDelta(0, "", "", `"Paris"}`)))
	complete := llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(10, 5, 15))

	if err := emitter.EmitFirst(&buf, chunk1); err != nil {
		t.Fatalf("EmitFirst: %v", err)
	}
	if err := emitter.Emit(&buf, chunk2); err != nil {
		t.Fatalf("Emit chunk2: %v", err)
	}
	if err := emitter.Emit(&buf, chunk3); err != nil {
		t.Fatalf("Emit chunk3: %v", err)
	}
	if err := emitter.Emit(&buf, complete); err != nil {
		t.Fatalf("Emit complete: %v", err)
	}
	if err := emitter.Finalize(&buf, complete.Usage()); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	events := parseSSEEvents(t, buf.String())

	wantTypes := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}
	if len(events) != len(wantTypes) {
		t.Fatalf("events = %d, want %d: %+v", len(events), len(wantTypes), events)
	}
	for i, want := range wantTypes {
		if events[i].Event != want {
			t.Errorf("event[%d] = %q, want %q", i, events[i].Event, want)
		}
	}

	contentBlock := events[1].Data["content_block"].(map[string]any)
	if contentBlock["type"] != "tool_use" {
		t.Errorf("content_block.type = %v, want tool_use", contentBlock["type"])
	}
	if contentBlock["id"] != "toolu_1" || contentBlock["name"] != "get_weather" {
		t.Errorf("content_block = %+v", contentBlock)
	}

	delta1 := events[2].Data["delta"].(map[string]any)
	if delta1["type"] != "input_json_delta" || delta1["partial_json"] != `{"location":` {
		t.Errorf("delta1 = %+v", delta1)
	}
	delta2 := events[3].Data["delta"].(map[string]any)
	if delta2["type"] != "input_json_delta" || delta2["partial_json"] != `"Paris"}` {
		t.Errorf("delta2 = %+v", delta2)
	}

	msgDelta := events[5].Data["delta"].(map[string]any)
	if msgDelta["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use", msgDelta["stop_reason"])
	}
}

func TestAnthropicStreamEmitter_ThinkingTextToolUse(t *testing.T) {
	emitter := newAnthropicStreamEmitter("claude-3-5-sonnet-20241022")
	var buf bytes.Buffer

	thinkingChunk := llm.NewStreamChunk(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", "Let me think...", nil))
	textChunk := llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "The answer is 4."))
	toolChunk := llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "",
		llm.NewToolCallDelta(0, "toolu_1", "calculator", `{}`)))
	complete := llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(10, 5, 15))

	if err := emitter.EmitFirst(&buf, thinkingChunk); err != nil {
		t.Fatalf("EmitFirst: %v", err)
	}
	if err := emitter.Emit(&buf, textChunk); err != nil {
		t.Fatalf("Emit textChunk: %v", err)
	}
	if err := emitter.Emit(&buf, toolChunk); err != nil {
		t.Fatalf("Emit toolChunk: %v", err)
	}
	if err := emitter.Emit(&buf, complete); err != nil {
		t.Fatalf("Emit complete: %v", err)
	}
	if err := emitter.Finalize(&buf, complete.Usage()); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	events := parseSSEEvents(t, buf.String())

	wantTypes := []string{
		"message_start",
		"content_block_start", // thinking, index 0
		"content_block_delta", // thinking_delta
		"content_block_stop",  // close index 0
		"content_block_start", // text, index 1
		"content_block_delta", // text_delta
		"content_block_stop",  // close index 1
		"content_block_start", // tool_use, index 2
		"content_block_delta", // input_json_delta
		"content_block_stop",  // close index 2
		"message_delta",
		"message_stop",
	}
	if len(events) != len(wantTypes) {
		t.Fatalf("events = %d, want %d: %+v", len(events), len(wantTypes), events)
	}
	for i, want := range wantTypes {
		if events[i].Event != want {
			t.Errorf("event[%d] = %q, want %q", i, events[i].Event, want)
		}
	}

	thinkingBlock := events[1].Data["content_block"].(map[string]any)
	if thinkingBlock["type"] != "thinking" {
		t.Errorf("content_block.type = %v, want thinking", thinkingBlock["type"])
	}
	if events[1].Data["index"] != float64(0) {
		t.Errorf("thinking block index = %v, want 0", events[1].Data["index"])
	}

	thinkingDelta := events[2].Data["delta"].(map[string]any)
	if thinkingDelta["type"] != "thinking_delta" || thinkingDelta["thinking"] != "Let me think..." {
		t.Errorf("thinking delta = %+v", thinkingDelta)
	}

	textBlock := events[4].Data["content_block"].(map[string]any)
	if textBlock["type"] != "text" {
		t.Errorf("content_block.type = %v, want text", textBlock["type"])
	}
	if events[4].Data["index"] != float64(1) {
		t.Errorf("text block index = %v, want 1", events[4].Data["index"])
	}

	toolBlock := events[7].Data["content_block"].(map[string]any)
	if toolBlock["type"] != "tool_use" {
		t.Errorf("content_block.type = %v, want tool_use", toolBlock["type"])
	}
	if events[7].Data["index"] != float64(2) {
		t.Errorf("tool_use block index = %v, want 2", events[7].Data["index"])
	}

	msgDelta := events[10].Data["delta"].(map[string]any)
	if msgDelta["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use", msgDelta["stop_reason"])
	}
}

func TestAnthropicStreamEmitter_Error(t *testing.T) {
	emitter := newAnthropicStreamEmitter("claude-3-5-sonnet-20241022")
	var buf bytes.Buffer

	chunk1 := llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "Hello"))
	if err := emitter.EmitFirst(&buf, chunk1); err != nil {
		t.Fatalf("EmitFirst: %v", err)
	}

	if err := emitter.EmitError(&buf, NewInternalError("boom")); err != nil {
		t.Fatalf("EmitError: %v", err)
	}

	events := parseSSEEvents(t, buf.String())
	last := events[len(events)-1]
	if last.Event != "error" {
		t.Fatalf("last event = %q, want error", last.Event)
	}
	errObj := last.Data["error"].(map[string]any)
	if errObj["message"] != "boom" {
		t.Errorf("error message = %v, want boom", errObj["message"])
	}
}

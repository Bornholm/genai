package proxy

import (
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
)

// marshalChunk renders a formatted chunk back to the generic JSON shape a
// client would actually parse off the wire.
func marshalChunk(t *testing.T, payload any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("could not marshal chunk: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("could not unmarshal chunk: %v", err)
	}
	return out
}

func firstChoice(t *testing.T, chunk map[string]any) map[string]any {
	t.Helper()
	choices, ok := chunk["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("expected at least one choice, got %v", chunk["choices"])
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		t.Fatalf("unexpected choice shape %T", choices[0])
	}
	return choice
}

// Tool call arguments are streamed across several deltas. Emitting a
// finish_reason on an intermediate chunk makes OpenAI-compatible clients close
// the turn on truncated arguments and retry the call, which shows up as an
// agent looping on the same tools.
func TestFormatStreamChunkNoFinishReasonOnIntermediateToolCalls(t *testing.T) {
	deltas := []llm.StreamDelta{
		llm.NewStreamDelta(llm.RoleAssistant, "", llm.NewToolCallDelta(0, "call_1", "read_file", `{"pa`)),
		llm.NewStreamDelta(llm.RoleAssistant, "", llm.NewToolCallDelta(0, "", "", `th":"/tmp/x"}`)),
	}

	for i, d := range deltas {
		chunk := marshalChunk(t, FormatStreamChunk(llm.NewStreamChunk(d), "id", "m", true))
		choice := firstChoice(t, chunk)
		if fr := choice["finish_reason"]; fr != nil {
			t.Errorf("delta chunk %d: finish_reason = %v, want nil", i, fr)
		}
	}
}

// The terminal chunk carries finish_reason:"tool_calls" so the client knows the
// tool-call turn is complete.
func TestFormatStreamChunkTerminalFinishReason(t *testing.T) {
	for _, tc := range []struct {
		name         string
		sawToolCalls bool
		want         string
	}{
		{"tool call stream", true, "tool_calls"},
		{"text stream", false, "stop"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			usage := llm.NewChatCompletionUsage(10, 20, 30)
			chunk := marshalChunk(t, FormatStreamChunk(llm.NewCompleteStreamChunk(usage), "id", "m", tc.sawToolCalls))

			choice := firstChoice(t, chunk)
			if got := choice["finish_reason"]; got != tc.want {
				t.Errorf("finish_reason = %v, want %q", got, tc.want)
			}
			if chunk["usage"] == nil {
				t.Error("terminal chunk should carry usage")
			}
		})
	}
}

// The provider's tool call index must survive: clients key their accumulator on
// it, so renumbering per chunk merges parallel calls and concatenates their
// arguments into invalid JSON.
func TestFormatStreamChunkPreservesToolCallIndex(t *testing.T) {
	// Second parallel call, arriving alone in its own chunk — position 0 in the
	// slice, but index 1 in the response.
	delta := llm.NewStreamDelta(llm.RoleAssistant, "", llm.NewToolCallDelta(1, "call_2", "write_file", `{}`))

	chunk := marshalChunk(t, FormatStreamChunk(llm.NewStreamChunk(delta), "id", "m", true))
	choice := firstChoice(t, chunk)

	toolCalls := choice["delta"].(map[string]any)["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if got := toolCalls[0].(map[string]any)["index"]; got != float64(1) {
		t.Errorf("index = %v, want 1", got)
	}
}

func TestFormatStreamChunkForwardsReasoningDetails(t *testing.T) {
	details := []llm.ReasoningDetail{{
		ID: "r1", Type: llm.ReasoningDetailTypeText, Text: "thinking", Format: "minimax", Index: 0,
	}}
	delta := llm.NewReasoningStreamDelta(llm.RoleAssistant, "", "thinking", details)

	chunk := marshalChunk(t, FormatStreamChunk(llm.NewStreamChunk(delta), "id", "m", false))
	got := firstChoice(t, chunk)["delta"].(map[string]any)["reasoning_details"]

	blocks, ok := got.([]any)
	if !ok || len(blocks) != 1 {
		t.Fatalf("reasoning_details = %v, want 1 block", got)
	}
	if id := blocks[0].(map[string]any)["id"]; id != "r1" {
		t.Errorf("reasoning detail id = %v, want r1", id)
	}
}

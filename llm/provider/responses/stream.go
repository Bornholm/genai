package responses

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// streamEvent is one event of a response stream, reduced to the fields
// genai reads. Each event type fills a few of them.
type streamEvent struct {
	Type         string         `json:"type"`
	OutputIndex  int            `json:"output_index"`
	SummaryIndex int            `json:"summary_index"`
	ContentIndex int            `json:"content_index"`
	Delta        string         `json:"delta"`
	Item         *outputItem    `json:"item"`
	Response     *response      `json:"response"`
	Code         string         `json:"code"`
	Message      string         `json:"message"`
	Error        *responseError `json:"error"`
}

// reasoningPart identifies one piece of readable reasoning: a summary part
// or a raw reasoning text part of a reasoning item.
type reasoningPart struct {
	outputIndex int
	index       int
	raw         bool
}

// streamEmitter translates Responses API events into genai stream chunks.
//
// Readable reasoning (summaries, or raw reasoning text) streams as reasoning
// deltas, separated by a blank line as fromResponse joins them. The details
// that replay a reasoning item, encrypted content included, are emitted once
// the item is done.
//
// The API streams text and arguments as deltas. A gateway may instead ship
// them whole on the item events; the emitter then takes them from there,
// once.
type streamEmitter struct {
	ctx              context.Context
	chunks           chan<- llm.StreamChunk
	excludeReasoning bool

	// toolIndexes maps an output index to the tool call index genai exposes,
	// which counts tool calls only.
	toolIndexes map[int]int
	// argumentsSent marks the function calls whose arguments were emitted,
	// whether on the added event or as deltas.
	argumentsSent map[int]bool
	// argumentsWhole marks the function calls whose added event carried
	// their whole arguments: deltas repeating them would double the JSON.
	argumentsWhole map[int]bool
	// textSent marks the message items whose text was streamed.
	textSent map[int]bool
	// reasoningSent marks the reasoning items whose readable text was streamed.
	reasoningSent map[int]bool

	lastReasoning *reasoningPart
	detailCount   int
	// answered is set once the output holds a message or a function call.
	answered bool

	usage *responseUsage
}

func newStreamEmitter(ctx context.Context, chunks chan<- llm.StreamChunk, excludeReasoning bool) *streamEmitter {
	return &streamEmitter{
		ctx:              ctx,
		chunks:           chunks,
		excludeReasoning: excludeReasoning,
		toolIndexes:      map[int]int{},
		argumentsSent:    map[int]bool{},
		argumentsWhole:   map[int]bool{},
		textSent:         map[int]bool{},
		reasoningSent:    map[int]bool{},
	}
}

// handle processes the data of one event and reports whether it ended the
// stream.
func (e *streamEmitter) handle(data []byte) bool {
	var event streamEvent
	if err := json.Unmarshal(data, &event); err != nil {
		e.fail(errors.WithStack(llm.RateLimitError(http.StatusBadGateway, "malformed stream event: "+err.Error())))
		return true
	}

	switch event.Type {
	case "response.output_item.added":
		if event.Item == nil {
			return false
		}
		switch event.Item.Type {
		case "function_call":
			e.answered = true
			index := e.toolIndex(event.OutputIndex)
			arguments := event.Item.Arguments
			if arguments != "" {
				e.argumentsSent[event.OutputIndex] = true
				e.argumentsWhole[event.OutputIndex] = true
			}
			e.send(llm.NewStreamDelta(llm.RoleAssistant, "",
				llm.NewToolCallDelta(index, event.Item.CallID, event.Item.Name, arguments),
			))
		}

	case "response.output_text.delta", "response.refusal.delta":
		if event.Delta != "" {
			// A gateway may stream a message without announcing its item:
			// text sent is an answer, whatever comes next. An empty delta
			// sends nothing, so the finished item may still carry the text.
			e.textSent[event.OutputIndex] = true
			e.answered = true
			e.send(llm.NewStreamDelta(llm.RoleAssistant, event.Delta))
		}

	case "response.function_call_arguments.delta":
		index, known := e.toolIndexes[event.OutputIndex]
		if !known || event.Delta == "" || e.argumentsWhole[event.OutputIndex] {
			return false
		}
		e.argumentsSent[event.OutputIndex] = true
		e.send(llm.NewStreamDelta(llm.RoleAssistant, "",
			llm.NewToolCallDelta(index, "", "", event.Delta),
		))

	case "response.reasoning_summary_text.delta":
		e.reasoning(reasoningPart{outputIndex: event.OutputIndex, index: event.SummaryIndex}, event.Delta)

	case "response.reasoning_text.delta":
		e.reasoning(reasoningPart{outputIndex: event.OutputIndex, index: event.ContentIndex, raw: true}, event.Delta)

	case "response.output_item.done":
		if event.Item != nil {
			e.itemDone(event.OutputIndex, *event.Item)
		}

	case "response.completed", "response.incomplete":
		e.recordUsage(event.Response)
		if !e.answered {
			e.fail(errors.WithStack(llm.ErrNoMessage))
			return true
		}
		e.sendTerminal(llm.NewCompleteStreamChunk(newUsage(e.usage)))
		return true

	case "response.failed":
		e.recordUsage(event.Response)
		var err error
		if event.Response != nil {
			event.Response.Status = "failed"
			err = responseFailure(event.Response)
		} else {
			err = failureError("", "the response failed")
		}
		e.fail(errors.WithStack(err))
		return true

	case "error":
		code, message := event.Code, event.Message
		if event.Error != nil {
			code, message = event.Error.Code, event.Error.Message
		}
		e.fail(errors.WithStack(failureError(code, message)))
		return true

	default:
		// response.created and response.in_progress carry the response as
		// it stands, without usage on the API but possibly with it on a
		// gateway.
		e.recordUsage(event.Response)
	}

	return false
}

// itemDone takes from a finished item what its deltas did not carry, and
// emits the details of a reasoning item.
func (e *streamEmitter) itemDone(outputIndex int, item outputItem) {
	switch item.Type {
	case "message":
		// A message without text is no answer: the stream then ends on
		// ErrNoMessage, which a retry handles.
		if !e.textSent[outputIndex] {
			e.textSent[outputIndex] = true
			if text := messageText(item); text != "" {
				e.answered = true
				e.send(llm.NewStreamDelta(llm.RoleAssistant, text))
			}
		}

	case "function_call":
		e.answered = true
		index, known := e.toolIndexes[outputIndex]
		if !known {
			index = e.toolIndex(outputIndex)
			e.argumentsSent[outputIndex] = true
			e.send(llm.NewStreamDelta(llm.RoleAssistant, "",
				llm.NewToolCallDelta(index, item.CallID, item.Name, item.Arguments),
			))
			return
		}
		if !e.argumentsSent[outputIndex] && item.Arguments != "" {
			e.argumentsSent[outputIndex] = true
			e.send(llm.NewStreamDelta(llm.RoleAssistant, "",
				llm.NewToolCallDelta(index, "", "", item.Arguments),
			))
		}

	case "reasoning":
		if !e.reasoningSent[outputIndex] {
			// No delta came for this item: its text, if any, is here.
			if text := reasoningText(item); text != "" {
				e.reasoning(reasoningPart{outputIndex: outputIndex}, text)
			}
		}
		details := reasoningDetails(item, e.detailCount)
		if len(details) == 0 {
			return
		}
		e.detailCount += len(details)
		e.send(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", "", details))
	}
}

// reasoning emits one delta of readable reasoning, opening a new paragraph
// when it starts another part.
func (e *streamEmitter) reasoning(part reasoningPart, delta string) {
	if delta == "" {
		return
	}
	e.reasoningSent[part.outputIndex] = true
	if e.excludeReasoning {
		return
	}
	if e.lastReasoning != nil && *e.lastReasoning != part {
		delta = "\n\n" + delta
	}
	e.lastReasoning = &part
	e.send(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", delta, nil))
}

func (e *streamEmitter) toolIndex(outputIndex int) int {
	index, known := e.toolIndexes[outputIndex]
	if !known {
		index = len(e.toolIndexes)
		e.toolIndexes[outputIndex] = index
	}
	return index
}

func (e *streamEmitter) recordUsage(res *response) {
	if res != nil && usagePublished(res.Usage) {
		e.usage = res.Usage
	}
}

// send emits one delta chunk, carrying the usage known so far once the
// provider has published any.
func (e *streamEmitter) send(delta llm.StreamDelta) {
	if e.usage != nil {
		llm.SendChunk(e.ctx, e.chunks, llm.NewStreamChunkWithUsage(delta, newUsage(e.usage)))
		return
	}
	llm.SendChunk(e.ctx, e.chunks, llm.NewStreamChunk(delta))
}

func (e *streamEmitter) sendTerminal(chunk llm.StreamChunk) {
	llm.SendTerminalChunk(e.ctx, e.chunks, chunk)
}

// fail ends the stream on err, with the usage published so far if any: a
// zeroed usage would read as "this cost nothing" rather than "unknown".
func (e *streamEmitter) fail(err error) {
	if e.usage != nil {
		e.sendTerminal(llm.NewErrorStreamChunkWithUsage(err, newUsage(e.usage)))
		return
	}
	e.sendTerminal(llm.NewErrorStreamChunk(err))
}

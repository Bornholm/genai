package gemini

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// streamEmitter translates the chunks of a streamGenerateContent stream into
// genai stream chunks. Each chunk carries new parts; a function call comes
// whole in one part. The last chunk carries the finish reason, and the
// stream then ends without a terminal event: a stream ending without a
// finish reason was cut.
type streamEmitter struct {
	ctx              context.Context
	chunks           chan<- llm.StreamChunk
	excludeReasoning bool

	toolCount   int
	detailCount int
	answered    bool
	finish      string

	usage *usageMetadata
}

func newStreamEmitter(ctx context.Context, chunks chan<- llm.StreamChunk, excludeReasoning bool) *streamEmitter {
	return &streamEmitter{ctx: ctx, chunks: chunks, excludeReasoning: excludeReasoning}
}

// handle processes the data of one event and reports whether it ended the
// stream.
func (e *streamEmitter) handle(data []byte) bool {
	var chunk response
	if err := json.Unmarshal(data, &chunk); err != nil {
		e.fail(errors.WithStack(llm.RateLimitError(http.StatusBadGateway, "malformed stream chunk: "+err.Error())))
		return true
	}

	if chunk.Error != nil {
		e.fail(errors.WithStack(streamError(chunk.Error)))
		return true
	}
	if usagePublished(chunk.UsageMetadata) {
		e.usage = chunk.UsageMetadata
	}
	if chunk.PromptFeedback != nil && chunk.PromptFeedback.BlockReason != "" {
		e.fail(errors.WithStack(blockedError("prompt", chunk.PromptFeedback.BlockReason)))
		return true
	}
	if len(chunk.Candidates) == 0 {
		return false
	}

	candidate := chunk.Candidates[0]
	if candidate.Content != nil {
		for _, p := range candidate.Content.Parts {
			e.part(p)
		}
	}
	if candidate.FinishReason != "" {
		e.finish = candidate.FinishReason
	}
	return false
}

func (e *streamEmitter) part(p part) {
	switch {
	case p.FunctionCall != nil:
		e.answered = true
		id := p.FunctionCall.ID
		if id == "" {
			id = newCallID()
		}
		args := callArguments(p.FunctionCall.Args)
		index := e.toolCount
		e.toolCount++
		e.send(llm.NewStreamDelta(llm.RoleAssistant, "",
			llm.NewToolCallDelta(index, id, p.FunctionCall.Name, string(args)),
		))
		e.signature(id, p.ThoughtSignature)
		return
	case p.Thought:
		// Thoughts are not replayed, so neither is their signature.
		if p.Text != "" && !e.excludeReasoning {
			e.send(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", p.Text, nil))
		}
	default:
		if p.Text != "" {
			e.answered = true
			e.send(llm.NewStreamDelta(llm.RoleAssistant, p.Text))
		}
		e.signature("", p.ThoughtSignature)
	}
}

// signature emits a thought signature as a reasoning detail.
func (e *streamEmitter) signature(id, signature string) {
	if signature == "" {
		return
	}
	detail := signatureDetail(id, signature, e.detailCount)
	e.detailCount++
	e.send(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", "", []llm.ReasoningDetail{detail}))
}

// end closes a stream the upstream ended.
func (e *streamEmitter) end() {
	switch {
	case e.finish == "":
		e.fail(errors.WithStack(llm.NewHTTPError(http.StatusBadGateway, "gemini: the stream ended before the model finished its answer")))
	case !finished(e.finish) && !e.answered:
		e.fail(errors.WithStack(blockedError("answer", e.finish)))
	case !e.answered:
		e.fail(errors.WithStack(llm.ErrNoMessage))
	default:
		// A safety stop after some text keeps what was sent, like an
		// OpenAI content filter.
		e.sendTerminal(llm.NewCompleteStreamChunk(newUsage(e.usage)))
	}
}

// send emits one delta chunk, carrying the usage known so far once the
// provider has published any: Gemini repeats it on every chunk.
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

// fail ends the stream on err, with the usage published so far if any.
func (e *streamEmitter) fail(err error) {
	if e.usage != nil {
		e.sendTerminal(llm.NewErrorStreamChunkWithUsage(err, newUsage(e.usage)))
		return
	}
	e.sendTerminal(llm.NewErrorStreamChunk(err))
}

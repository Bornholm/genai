package llm

// FinishReason tells why a model stopped generating. The values follow the
// OpenAI vocabulary, which the providers map theirs to; the empty value means
// the provider did not say.
type FinishReason string

const (
	// FinishReasonStop means the model ended its answer, or met a stop
	// sequence.
	FinishReasonStop FinishReason = "stop"
	// FinishReasonLength means the answer was cut by the token limit, the
	// request's or the model's.
	FinishReasonLength FinishReason = "length"
	// FinishReasonToolCalls means the model stopped to have tools called.
	FinishReasonToolCalls FinishReason = "tool_calls"
	// FinishReasonContentFilter means the provider withheld or cut the answer
	// for its content, a refusal included.
	FinishReasonContentFilter FinishReason = "content_filter"
)

// FinishReasonReporter is implemented by the chat completion responses and
// stream chunks that know why the model stopped. In a stream the reason
// usually travels on the completion chunk, but a provider may send it on an
// earlier one, before a trailing usage chunk: a consumer keeps the last
// non-empty reason it saw.
type FinishReasonReporter interface {
	FinishReason() FinishReason
}

// ParseOpenAIFinishReason reads a finish_reason of the OpenAI chat completion
// API, and the variants OpenAI-compatible providers send. A value it does not
// know comes back empty rather than as is: the proxy then infers one, instead
// of passing on a value OpenAI clients would not recognize.
func ParseOpenAIFinishReason(value string) FinishReason {
	switch value {
	case "stop", "end_turn", "stop_sequence":
		return FinishReasonStop
	case "length", "max_tokens", "model_length":
		return FinishReasonLength
	case "tool_calls", "function_call", "tool_use":
		return FinishReasonToolCalls
	case "content_filter", "refusal":
		return FinishReasonContentFilter
	default:
		return ""
	}
}

// FinishReasonOf returns the finish reason v reports, empty when it reports
// none. A decorator wrapping a response hides it unless it implements
// FinishReasonReporter itself.
func FinishReasonOf(v any) FinishReason {
	if r, ok := v.(FinishReasonReporter); ok {
		return r.FinishReason()
	}
	return ""
}

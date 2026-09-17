package proxy

import (
	"encoding/json"
	"net/http"

	"github.com/bornholm/genai/llm"
)

type RequestType string

const (
	RequestTypeChatCompletion RequestType = "chat_completion"
	RequestTypeEmbedding      RequestType = "embedding"
	RequestTypeModels         RequestType = "models"
	RequestTypeMessage        RequestType = "message"
	RequestTypeCountTokens    RequestType = "count_tokens"
)

// ProxyRequest encapsulates any request transiting through the proxy.
type ProxyRequest struct {
	Type    RequestType
	Model   string          // model requested by the client
	UserID  string          // extracted identity (API key, header, JWT…)
	Headers http.Header     // original headers
	Body    json.RawMessage // raw request body

	// For chat completions — populated after parsing
	ChatOptions []llm.ChatCompletionOptionFunc

	// For embeddings — populated after parsing
	EmbeddingOptions []llm.EmbeddingsOptionFunc

	// Mutable metadata hooks can enrich
	Metadata map[string]any
}

// ProxyResponse encapsulates the response before sending it to the client.
type ProxyResponse struct {
	StatusCode int
	Body       any // will be serialized to JSON in OpenAI format
	Headers    http.Header
	TokensUsed *TokenUsage // filled after the LLM call

	// Interruption describes how a streamed response ended when it did not end
	// normally. It is nil for a completed stream and for every non-streamed
	// response. Post-response hooks receive the response either way: the request
	// was made, the tokens the provider produced before the interruption were
	// billed to the platform and delivered to the client, so they still have to
	// be accounted for. Whether TokensUsed holds those tokens depends on the
	// provider — see StreamInterruption.PartialUsage.
	Interruption *StreamInterruption
}

// StreamInterruptionCause tells apart the ways a streamed response stops early.
// They are not symmetric: one is an upstream failure, one is ordinary client
// traffic, one is a provider going silent.
type StreamInterruptionCause string

const (
	// StreamInterruptionUpstream means the provider itself failed mid-stream,
	// after chunks had already reached the client. The error is forwarded to the
	// client as an SSE error event and carried in StreamInterruption.Err.
	StreamInterruptionUpstream StreamInterruptionCause = "upstream_error"
	// StreamInterruptionClientGone means writing to the client failed — a closed
	// tab, an aborted request, a reverse proxy timing out. The upstream stream is
	// abandoned at that point.
	StreamInterruptionClientGone StreamInterruptionCause = "client_gone"
	// StreamInterruptionTruncated means the provider closed the stream without
	// ever signalling completion and without reporting an error. The client is
	// sent the normal closing events all the same — a provider ending a
	// legitimate response without a terminal chunk is not a protocol error, and
	// an error event would let the client retry a response it fully received —
	// so this cause is for accounting: what the exchange cost is unknown,
	// PartialUsage says so, and an upstream connection dropped cleanly looks
	// exactly like this. An upstream that goes silent without closing its
	// channel is a different matter and is not detected here: the read loop has
	// no inactivity deadline, because no timeout distinguishes a stalled
	// provider from a long, legitimate generation.
	StreamInterruptionTruncated StreamInterruptionCause = "stream_truncated"
)

// StreamInterruption records why a streamed response stopped before the
// provider signalled completion.
type StreamInterruption struct {
	Cause StreamInterruptionCause
	// Err is the underlying error. It is the provider error for
	// StreamInterruptionUpstream and the failed write for
	// StreamInterruptionClientGone.
	Err error
	// ChunksEmitted is how many content chunks were written to the client
	// before the interruption, the first one included; error and closing events
	// are not counted. It is the only measure of the volume produced that is
	// always available — see PartialUsage.
	ChunksEmitted int
	// TerminalEventUndelivered marks an exchange whose last SSE event never
	// reached the client: the error event of an upstream failure, or the
	// closing events of a stream that ended otherwise. It is always true on a
	// client hangup, where nothing more could be written by definition. Cause
	// keeps saying what stopped the stream — an upstream failure stays
	// StreamInterruptionUpstream even when the client had gone away too — and
	// this says the client was not there to read how it ended.
	TerminalEventUndelivered bool
	// PartialUsage reports whether ProxyResponse.TokensUsed holds counts the
	// provider actually published before the interruption.
	//
	// It is false far more often than one would hope: most providers only
	// report usage in the final chunk of a stream, which by definition never
	// arrives here. Anthropic publishes the input tokens in message_start and
	// the output tokens as the stream runs, so its counts are real; the
	// OpenAI-compatible providers usually report nothing until the end, and
	// TokensUsed is then entirely zero. A hook must check this flag before
	// charging an interrupted stream: a zeroed TokensUsed means "unknown", not
	// "free". ChunksEmitted remains as a volume proxy.
	PartialUsage bool
}

type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	CachedTokens     int
	Cost             *float64 // provider-reported cost, nil if not available
	CostCurrency     string
}

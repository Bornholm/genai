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
	// response. Post-response hooks receive the response either way: the tokens
	// the provider produced before the interruption were billed to the platform
	// and delivered to the client, so they still have to be accounted for.
	Interruption *StreamInterruption
}

// StreamInterruptionCause tells apart the two ways a streamed response stops
// early. They are not symmetric: one is an upstream failure, the other ordinary
// client traffic.
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
)

// StreamInterruption records why a streamed response stopped before the
// provider signalled completion.
type StreamInterruption struct {
	Cause StreamInterruptionCause
	// Err is the underlying error. It is the provider error for
	// StreamInterruptionUpstream and the failed write for
	// StreamInterruptionClientGone.
	Err error
	// ChunksEmitted is how many chunks were written to the client before the
	// interruption, the first one included.
	ChunksEmitted int
}

type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	CachedTokens     int
	Cost             *float64 // provider-reported cost, nil if not available
	CostCurrency     string
}

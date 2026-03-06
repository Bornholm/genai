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
	Body       any         // will be serialized to JSON in OpenAI format
	Headers    http.Header
	TokensUsed *TokenUsage // filled after the LLM call
}

type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

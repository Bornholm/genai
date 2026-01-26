package llm

type Client interface {
	ChatCompletionClient
	ChatCompletionStreamingClient
	EmbeddingsClient
}

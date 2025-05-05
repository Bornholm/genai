package llm

type Client interface {
	ChatCompletionClient
	EmbeddingsClient
}

package yzma

import "github.com/pkg/errors"

// ChatCompletionOptions contient les options de configuration du provider yzma pour le chat.
type ChatCompletionOptions struct {
	ModelPath       string  `env:"MODEL_PATH"`
	ModelURL        string  `env:"MODEL_URL"`
	LibPath         string  `env:"LIB_PATH"`
	Processor       string  `env:"PROCESSOR"`
	Version         string  `env:"VERSION"`
	ContextSize     int     `env:"CONTEXT_SIZE"`
	BatchSize       int     `env:"BATCH_SIZE"`
	UBatchSize      int     `env:"U_BATCH_SIZE"`
	Temperature     float64 `env:"TEMPERATURE"`
	TopK            int     `env:"TOP_K"`
	TopP            float64 `env:"TOP_P"`
	MinP            float64 `env:"MIN_P"`
	PresencePenalty float64 `env:"PRESENCE_PENALTY"`
	PenaltyLastN    int     `env:"PENALTY_LAST_N"`
	PredictSize     int     `env:"PREDICT_SIZE"`
	Template        string  `env:"TEMPLATE"`
	Verbose         bool    `env:"VERBOSE"`
}

func defaultChatCompletionOptions() *ChatCompletionOptions {
	return &ChatCompletionOptions{
		ContextSize:     40960,
		BatchSize:       512,
		UBatchSize:      512,
		Temperature:     1.0,
		TopK:            20,
		TopP:            1.0,
		MinP:            0.0,
		PresencePenalty: 2.0,
		PenaltyLastN:    64,
		PredictSize:     32768,
	}
}

// Validate vérifie que les options minimales sont présentes.
func (o *ChatCompletionOptions) Validate() error {
	if o.ModelPath == "" && o.ModelURL == "" {
		return errors.New("field \"ModelPath\": model path or model URL is required")
	}
	return nil
}

// EmbeddingsOptions contient les options de configuration du provider yzma pour les embeddings.
type EmbeddingsOptions struct {
	ModelPath   string `env:"MODEL_PATH"`
	ModelURL    string `env:"MODEL_URL"`
	LibPath     string `env:"LIB_PATH"`
	Processor   string `env:"PROCESSOR"`
	Version     string `env:"VERSION"`
	ContextSize int    `env:"CONTEXT_SIZE"`
	BatchSize   int    `env:"BATCH_SIZE"`
	PoolingType int    `env:"POOLING_TYPE"`
	Normalize   bool   `env:"NORMALIZE"`
	Verbose     bool   `env:"VERBOSE"`
}

func defaultEmbeddingsOptions() *EmbeddingsOptions {
	return &EmbeddingsOptions{
		ContextSize: 40960,
		BatchSize:   512,
		Normalize:   true, // conserver le comportement par défaut de NewEmbeddingsClient
	}
}

// Validate vérifie que les options minimales sont présentes.
func (o *EmbeddingsOptions) Validate() error {
	if o.ModelPath == "" && o.ModelURL == "" {
		return errors.New("field \"ModelPath\": model path or model URL is required")
	}
	return nil
}

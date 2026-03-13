package provider_test

import (
	"context"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
)

// testChatOptions est une struct d'options de test avec un default et un validateur.
type testChatOptions struct {
	Model   string `env:"MODEL"`
	Timeout int    `env:"TIMEOUT"`
}

func (o *testChatOptions) Validate() error {
	if o.Model == "" {
		return llm.NewValidationError("model", "model is required")
	}
	return nil
}

// testEmbeddingsOptions est une struct d'options de test pour les embeddings.
type testEmbeddingsOptions struct {
	Model string `env:"MODEL"`
}

func TestRegisterChatCompletion_NewProviderOptions(t *testing.T) {
	const testProvider provider.Name = "test-chat-options-provider"

	provider.RegisterChatCompletion(
		testProvider,
		func() *testChatOptions {
			return &testChatOptions{Model: "default-model", Timeout: 30}
		},
		func(ctx context.Context, opts *testChatOptions) (llm.ChatCompletionClient, error) {
			return nil, nil
		},
	)

	opts := provider.NewChatCompletionProviderOptions(testProvider)
	if opts == nil {
		t.Fatal("expected non-nil options")
	}

	typed, ok := opts.(*testChatOptions)
	if !ok {
		t.Fatalf("expected *testChatOptions, got %T", opts)
	}

	if typed.Model != "default-model" {
		t.Errorf("expected default model 'default-model', got %q", typed.Model)
	}
	if typed.Timeout != 30 {
		t.Errorf("expected default timeout 30, got %d", typed.Timeout)
	}
}

func TestNewChatCompletionProviderOptions_UnknownProvider(t *testing.T) {
	opts := provider.NewChatCompletionProviderOptions("unknown-provider-xyz")
	if opts != nil {
		t.Errorf("expected nil for unknown provider, got %v", opts)
	}
}

func TestCreate_ErrNotConfigured_WhenNilOptions(t *testing.T) {
	ctx := context.Background()
	_, err := provider.Create(ctx) // aucune OptionFunc
	if err == nil {
		t.Fatal("expected error when no options provided")
	}
}

func TestCreate_ErrClientNotFound_UnknownProvider(t *testing.T) {
	ctx := context.Background()
	_, err := provider.Create(ctx, func(opts *provider.Options) error {
		opts.ChatCompletion = &provider.ResolvedClientOptions{
			Provider: "no-such-provider-abc",
			Specific: nil,
		}
		return nil
	})
	if err == nil {
		t.Fatal("expected error for unknown provider")
	}
}

func TestCreate_CallsValidate_AndReturnsError(t *testing.T) {
	const testProvider provider.Name = "test-validate-provider"

	provider.RegisterChatCompletion(
		testProvider,
		func() *testChatOptions { return &testChatOptions{} }, // Model vide → Validate() échoue
		func(ctx context.Context, opts *testChatOptions) (llm.ChatCompletionClient, error) {
			return nil, nil
		},
	)

	ctx := context.Background()
	_, err := provider.Create(ctx, func(opts *provider.Options) error {
		opts.ChatCompletion = &provider.ResolvedClientOptions{
			Provider: testProvider,
			Specific: &testChatOptions{}, // Model vide
		}
		return nil
	})
	if err == nil {
		t.Fatal("expected validation error")
	}
}

func TestCreate_Success(t *testing.T) {
	const testProvider provider.Name = "test-success-provider"

	var receivedModel string
	provider.RegisterChatCompletion(
		testProvider,
		func() *testChatOptions { return &testChatOptions{Model: "default"} },
		func(ctx context.Context, opts *testChatOptions) (llm.ChatCompletionClient, error) {
			receivedModel = opts.Model
			return &dummyChatClient{}, nil
		},
	)

	ctx := context.Background()
	_, err := provider.Create(ctx, func(opts *provider.Options) error {
		opts.ChatCompletion = &provider.ResolvedClientOptions{
			Provider: testProvider,
			Specific: &testChatOptions{Model: "gpt-test"},
		}
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if receivedModel != "gpt-test" {
		t.Errorf("expected model 'gpt-test', got %q", receivedModel)
	}
}

// dummyChatClient implémente llm.ChatCompletionClient pour les tests.
type dummyChatClient struct{}

func (d *dummyChatClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return nil, nil
}

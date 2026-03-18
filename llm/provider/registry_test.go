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

func TestWithChatCompletion_SetsCorrectProvider(t *testing.T) {
	optFunc := provider.WithChatCompletion("test-provider", testChatOptions{Model: "test-model"})

	opts := &provider.Options{}
	if err := optFunc(opts); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if opts.ChatCompletion == nil {
		t.Fatal("expected ChatCompletion to be set")
	}

	if opts.ChatCompletion.Provider != "test-provider" {
		t.Errorf("expected provider 'test-provider', got %q", opts.ChatCompletion.Provider)
	}

	specific, ok := opts.ChatCompletion.Specific.(*testChatOptions)
	if !ok {
		t.Fatalf("expected *testChatOptions, got %T", opts.ChatCompletion.Specific)
	}

	if specific.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", specific.Model)
	}
}

func TestWithEmbeddings_SetsCorrectProvider(t *testing.T) {
	optFunc := provider.WithEmbeddings("test-provider", testEmbeddingsOptions{Model: "test-embeddings-model"})

	opts := &provider.Options{}
	if err := optFunc(opts); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if opts.Embeddings == nil {
		t.Fatal("expected Embeddings to be set")
	}

	if opts.Embeddings.Provider != "test-provider" {
		t.Errorf("expected provider 'test-provider', got %q", opts.Embeddings.Provider)
	}

	specific, ok := opts.Embeddings.Specific.(*testEmbeddingsOptions)
	if !ok {
		t.Fatalf("expected *testEmbeddingsOptions, got %T", opts.Embeddings.Specific)
	}

	if specific.Model != "test-embeddings-model" {
		t.Errorf("expected model 'test-embeddings-model', got %q", specific.Model)
	}
}

func TestWithChatCompletion_CreatesClientSuccessfully(t *testing.T) {
	const testProvider provider.Name = "test-with-chat-provider"

	var receivedOpts *testChatOptions
	provider.RegisterChatCompletion(
		testProvider,
		func() *testChatOptions { return &testChatOptions{Model: "default"} },
		func(ctx context.Context, opts *testChatOptions) (llm.ChatCompletionClient, error) {
			receivedOpts = opts
			return &dummyChatClient{}, nil
		},
	)

	ctx := context.Background()
	client, err := provider.Create(ctx, provider.WithChatCompletion(testProvider, testChatOptions{
		Model:   "gpt-4",
		Timeout: 60,
	}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if client == nil {
		t.Fatal("expected non-nil client")
	}

	if receivedOpts.Model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got %q", receivedOpts.Model)
	}
	if receivedOpts.Timeout != 60 {
		t.Errorf("expected timeout 60, got %d", receivedOpts.Timeout)
	}
}

func TestWithEmbeddings_CreatesClientSuccessfully(t *testing.T) {
	const testProvider provider.Name = "test-with-embeddings-provider"

	// Register embeddings provider (reusing testChatOptions for simplicity)
	provider.RegisterEmbeddings(
		testProvider,
		func() *testChatOptions { return &testChatOptions{Model: "default"} },
		func(ctx context.Context, opts *testChatOptions) (llm.EmbeddingsClient, error) {
			return &dummyEmbeddingsClient{model: opts.Model}, nil
		},
	)

	ctx := context.Background()
	client, err := provider.Create(ctx, provider.WithEmbeddings(testProvider, testChatOptions{
		Model: "text-embedding-3-small",
	}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

// dummyEmbeddingsClient implements llm.EmbeddingsClient for tests.
type dummyEmbeddingsClient struct {
	model string
}

func (d *dummyEmbeddingsClient) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

func TestWithChatCompletion_Immutability(t *testing.T) {
	original := testChatOptions{Model: "original", Timeout: 30}
	optFunc := provider.WithChatCompletion("test-provider", original)

	// Modify the original
	original.Model = "modified"
	original.Timeout = 60

	opts := &provider.Options{}
	if err := optFunc(opts); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	specific := opts.ChatCompletion.Specific.(*testChatOptions)
	if specific.Model != "original" {
		t.Errorf("expected model 'original' (immutability), got %q", specific.Model)
	}
	if specific.Timeout != 30 {
		t.Errorf("expected timeout 30 (immutability), got %d", specific.Timeout)
	}
}

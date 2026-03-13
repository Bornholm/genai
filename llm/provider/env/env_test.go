package env_test

import (
	"context"
	"os"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	providerenv "github.com/bornholm/genai/llm/provider/env"
)

// envTestOptions est une struct d'options de test enregistrée pour ce test.
type envTestOptions struct {
	Model   string `env:"MODEL"`
	BaseURL string `env:"BASE_URL"`
	APIKey  string `env:"API_KEY"`
	Timeout int    `env:"TIMEOUT"`
}

func init() {
	provider.RegisterChatCompletion(
		"envtest",
		func() *envTestOptions {
			return &envTestOptions{BaseURL: "http://default.example.com", Timeout: 10}
		},
		func(ctx context.Context, opts *envTestOptions) (llm.ChatCompletionClient, error) {
			return nil, nil
		},
	)
	provider.RegisterEmbeddings(
		"envtest",
		func() *envTestOptions {
			return &envTestOptions{BaseURL: "http://default-emb.example.com"}
		},
		func(ctx context.Context, opts *envTestOptions) (llm.EmbeddingsClient, error) {
			return nil, nil
		},
	)
}

func TestWith_ParsesChatCompletionOptions(t *testing.T) {
	os.Setenv("TEST_CHAT_COMPLETION_PROVIDER", "envtest")
	os.Setenv("TEST_CHAT_COMPLETION_ENVTEST_MODEL", "my-model")
	os.Setenv("TEST_CHAT_COMPLETION_ENVTEST_API_KEY", "secret")
	os.Setenv("TEST_CHAT_COMPLETION_ENVTEST_TIMEOUT", "42")
	defer func() {
		os.Unsetenv("TEST_CHAT_COMPLETION_PROVIDER")
		os.Unsetenv("TEST_CHAT_COMPLETION_ENVTEST_MODEL")
		os.Unsetenv("TEST_CHAT_COMPLETION_ENVTEST_API_KEY")
		os.Unsetenv("TEST_CHAT_COMPLETION_ENVTEST_TIMEOUT")
	}()

	opts, err := provider.NewOptions(providerenv.With("TEST_"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if opts.ChatCompletion == nil {
		t.Fatal("expected ChatCompletion to be set")
	}
	if opts.ChatCompletion.Provider != "envtest" {
		t.Errorf("expected provider 'envtest', got %q", opts.ChatCompletion.Provider)
	}

	typed, ok := opts.ChatCompletion.Specific.(*envTestOptions)
	if !ok {
		t.Fatalf("expected *envTestOptions, got %T", opts.ChatCompletion.Specific)
	}
	if typed.Model != "my-model" {
		t.Errorf("expected model 'my-model', got %q", typed.Model)
	}
	if typed.APIKey != "secret" {
		t.Errorf("expected api key 'secret', got %q", typed.APIKey)
	}
	if typed.Timeout != 42 {
		t.Errorf("expected timeout 42, got %d", typed.Timeout)
	}
	// La valeur par défaut doit être préservée pour les champs non définis dans l'env
	if typed.BaseURL != "http://default.example.com" {
		t.Errorf("expected default base URL, got %q", typed.BaseURL)
	}
}

func TestWith_UnknownProvider_SilentlyIgnoresSpecificOpts(t *testing.T) {
	// Nom de provider sans tiret pour éviter les problèmes de variable d'env POSIX.
	os.Setenv("TEST2_CHAT_COMPLETION_PROVIDER", "nosuchprovider")
	os.Setenv("TEST2_CHAT_COMPLETION_NOSUCHPROVIDER_MODEL", "whatever")
	defer func() {
		os.Unsetenv("TEST2_CHAT_COMPLETION_PROVIDER")
		os.Unsetenv("TEST2_CHAT_COMPLETION_NOSUCHPROVIDER_MODEL")
	}()

	opts, err := provider.NewOptions(providerenv.With("TEST2_"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if opts.ChatCompletion == nil {
		t.Fatal("expected ChatCompletion to be set (provider identified even if unknown)")
	}
	if opts.ChatCompletion.Specific != nil {
		t.Errorf("expected nil Specific for unknown provider, got %v", opts.ChatCompletion.Specific)
	}
}

func TestWith_NoChatCompletionProvider_LeavesNil(t *testing.T) {
	// Aucune variable d'env avec le préfixe TEST3_
	opts, err := provider.NewOptions(providerenv.With("TEST3_"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if opts.ChatCompletion != nil {
		t.Errorf("expected nil ChatCompletion when no provider env var set")
	}
}

func TestWith_ParsesBothChatAndEmbeddings(t *testing.T) {
	os.Setenv("TEST4_CHAT_COMPLETION_PROVIDER", "envtest")
	os.Setenv("TEST4_CHAT_COMPLETION_ENVTEST_MODEL", "chat-model")
	os.Setenv("TEST4_EMBEDDINGS_PROVIDER", "envtest")
	os.Setenv("TEST4_EMBEDDINGS_ENVTEST_MODEL", "emb-model")
	defer func() {
		os.Unsetenv("TEST4_CHAT_COMPLETION_PROVIDER")
		os.Unsetenv("TEST4_CHAT_COMPLETION_ENVTEST_MODEL")
		os.Unsetenv("TEST4_EMBEDDINGS_PROVIDER")
		os.Unsetenv("TEST4_EMBEDDINGS_ENVTEST_MODEL")
	}()

	opts, err := provider.NewOptions(providerenv.With("TEST4_"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if opts.ChatCompletion == nil || opts.Embeddings == nil {
		t.Fatal("expected both ChatCompletion and Embeddings to be set")
	}

	chatTyped := opts.ChatCompletion.Specific.(*envTestOptions)
	embTyped := opts.Embeddings.Specific.(*envTestOptions)

	if chatTyped.Model != "chat-model" {
		t.Errorf("expected chat model 'chat-model', got %q", chatTyped.Model)
	}
	if embTyped.Model != "emb-model" {
		t.Errorf("expected embedding model 'emb-model', got %q", embTyped.Model)
	}
}

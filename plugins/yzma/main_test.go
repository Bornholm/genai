package main

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/plugin"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
)

// TestPluginStartsAndValidatesOptions builds the plugin and drives it through
// the host side, without a model: startup, capabilities and option validation
// are what can be checked everywhere.
func TestPluginStartsAndValidatesOptions(t *testing.T) {
	binary := filepath.Join(t.TempDir(), "genai-provider-yzma")
	build := exec.Command("go", "build", "-o", binary, ".")
	// Same setting as make build-plugins and the release build, so the
	// binary under test is the one that ships.
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		t.Fatalf("could not build plugin: %v", err)
	}
	defer plugin.CleanupClients()

	_, err := plugin.NewChatCompletionClient(context.Background(), binary, map[string]string{"LIB_PATH": "/nowhere"})
	if err == nil {
		t.Fatal("expected a validation error without a model")
	}
	var validationErr llm.ValidationError
	if !errors.As(err, &validationErr) {
		t.Errorf("expected the validation error to survive the plugin protocol, got %+v", err)
	}

	_, err = plugin.NewEmbeddingsClient(context.Background(), binary, nil)
	if err == nil {
		t.Fatal("expected a validation error without a model")
	}

	// One process serves the binary and reports both capabilities.
	proc, err := plugin.Start(context.Background(), binary)
	if err != nil {
		t.Fatalf("could not start plugin: %+v", err)
	}
	if got := proc.Info().GetCapabilities(); len(got) != 2 || got[0] != pluginv1.Capability_CAPABILITY_CHAT_COMPLETION || got[1] != pluginv1.Capability_CAPABILITY_EMBEDDINGS {
		t.Errorf("unexpected capabilities %v", got)
	}
}

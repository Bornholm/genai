package websearch

import (
	"context"
	"testing"

	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/openai"
	_ "github.com/bornholm/genai/llm/provider/openai"
	"github.com/davecgh/go-spew/spew"
	"github.com/pkg/errors"
	"github.com/testcontainers/testcontainers-go"
	tcollama "github.com/testcontainers/testcontainers-go/modules/ollama"
)

func TestTool(t *testing.T) {
	ctx := context.Background()

	t.Logf("Starting ollama container")

	ollamaContainer, err := tcollama.Run(ctx, "ollama/ollama:0.5.7", testcontainers.CustomizeRequest(testcontainers.GenericContainerRequest{
		ContainerRequest: testcontainers.ContainerRequest{
			Mounts: testcontainers.Mounts(
				testcontainers.VolumeMount("ollama-data", "/root/.ollama"),
			),
		},
	}))
	defer func() {
		if err := testcontainers.TerminateContainer(ollamaContainer); err != nil {
			t.Fatalf("failed to terminate container: %s", errors.WithStack(err))
		}
	}()
	if err != nil {
		t.Fatalf("failed to start container: %s", err)
	}

	model := "qwen2.5:3b"

	t.Logf("Pulling model '%s'", model)

	_, _, err = ollamaContainer.Exec(ctx, []string{"ollama", "pull", model})
	if err != nil {
		t.Fatalf("failed to pull model %s: %s", model, errors.WithStack(err))
	}

	connectionStr, err := ollamaContainer.ConnectionString(ctx)
	if err != nil {
		t.Fatalf("failed to get connection string: %s", errors.WithStack(err))
	}

	t.Logf("%s", connectionStr)

	client, err := provider.Create(ctx, provider.WithConfig(&provider.Config{
		Provider:            openai.Name,
		BaseURL:             connectionStr + "/v1/",
		ChatCompletionModel: model,
	}))

	tool := Tool(client)

	t.Logf("Executing tool")

	result, err := tool.Execute(ctx, map[string]any{"topic": "What are LLMs ?"})
	if err != nil {
		t.Fatalf("failed to get connection string: %s", errors.WithStack(err))
	}

	spew.Dump(result)

}

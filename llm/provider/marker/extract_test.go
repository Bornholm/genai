package marker

import (
	"context"
	"net/url"
	"testing"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/testsuite"
	"github.com/pkg/errors"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
)

func TestExtract(t *testing.T) {
	ctx := context.Background()
	req := testcontainers.ContainerRequest{
		Image:        "ghcr.io/bornholm/marker-pdf:1.6.2",
		ExposedPorts: []string{"8001/tcp"},
		Mounts: testcontainers.ContainerMounts{
			{
				Source: testcontainers.GenericVolumeMountSource{
					Name: "genai-marker-pdf-data",
				},
				Target: "/root/.cache",
			},
		},
		WaitingFor: wait.ForLog("Uvicorn running on http://0.0.0.0:8001").WithStartupTimeout(10 * time.Minute),
	}

	markerPDF, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: req,
		Started:          true,
	})
	defer testcontainers.CleanupContainer(t, markerPDF)

	if err != nil {
		t.Fatalf("+%v", errors.WithStack(err))
	}

	endpoint, err := markerPDF.Endpoint(ctx, "")
	if err != nil {
		t.Fatalf("+%v", errors.WithStack(err))
	}

	testsuite.ExtractText(t, func() (llm.ExtractTextClient, error) {
		baseURL := &url.URL{
			Scheme: "http",
			Host:   endpoint,
		}

		return NewExtractTextClient(baseURL), nil
	})
}

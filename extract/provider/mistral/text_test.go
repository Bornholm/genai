package mistral

import (
	"net/url"
	"os"
	"testing"

	"github.com/bornholm/genai/extract"
	"github.com/bornholm/genai/extract/testsuite"
)

func TestExtract(t *testing.T) {
	apiKey := os.Getenv("MISTRAL_API_KEY")

	if apiKey == "" {
		t.Skip("No MISTRAL_API_KEY environment variable, skipping")
		return
	}

	testsuite.ExtractText(t, func() (extract.TextClient, error) {
		baseURL := &url.URL{
			Scheme: "https",
			Host:   "api.mistral.ai",
		}

		return NewTextClient(baseURL, apiKey), nil
	})
}

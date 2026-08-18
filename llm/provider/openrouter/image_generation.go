package openrouter

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/pkg/errors"

	"github.com/bornholm/genai/llm"
)

// imageGenerationEndpoint est le point d'entrée dédié d'OpenRouter
// (POST /api/v1/images), distinct de /chat/completions : la génération
// d'images n'y passe pas par le dialecte "modalities" des complétions.
const imageGenerationEndpoint = "https://openrouter.ai/api/v1/images"

// ImageGenerationClient generates images through the OpenRouter images API.
//
// Le SDK openrouter utilisé pour les complétions ne couvre pas ce point
// d'entrée : le client HTTP est écrit ici directement, sur le format
// documenté (data[].b64_json + media_type).
type ImageGenerationClient struct {
	httpClient *http.Client
	endpoint   string
	apiKey     string
	model      string
}

type imageGenerationRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	N      int    `json:"n,omitempty"`
	Size   string `json:"size,omitempty"`
}

type imageGenerationResponse struct {
	Data []struct {
		B64JSON       string `json:"b64_json"`
		MediaType     string `json:"media_type"`
		RevisedPrompt string `json:"revised_prompt"`
	} `json:"data"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Code    any    `json:"code"`
	} `json:"error"`
}

// ImageGeneration implements [llm.ImageGenerationClient].
func (c *ImageGenerationClient) ImageGeneration(ctx context.Context, prompt string, funcs ...llm.ImageGenerationOptionFunc) (llm.ImageGenerationResponse, error) {
	opts := llm.NewImageGenerationOptions(funcs...)

	payload := imageGenerationRequest{
		Model:  c.model,
		Prompt: prompt,
		N:      opts.Count,
		Size:   opts.Size,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, errors.WithStack(err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 512<<20))
	if err != nil {
		return nil, errors.WithStack(err)
	}

	var parsed imageGenerationResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, errors.Wrapf(err, "unexpected response (HTTP %d)", resp.StatusCode)
	}

	if parsed.Error != nil {
		return nil, errors.Errorf("openrouter error (HTTP %d): %s", resp.StatusCode, parsed.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errors.Errorf("unexpected HTTP status %d", resp.StatusCode)
	}
	if len(parsed.Data) == 0 {
		return nil, errors.New("no image in response")
	}

	images := make([]llm.GeneratedImage, 0, len(parsed.Data))
	for _, img := range parsed.Data {
		// Certains modèles renvoient un data: URI complet plutôt que du
		// base64 nu ; les deux formes sont acceptées.
		b64 := img.B64JSON
		mediaType := img.MediaType
		if strings.HasPrefix(b64, "data:") {
			if idx := strings.Index(b64, ";base64,"); idx > 0 {
				if mediaType == "" {
					mediaType = b64[len("data:"):idx]
				}
				b64 = b64[idx+len(";base64,"):]
			}
		}

		data, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			return nil, errors.Wrap(err, "could not decode base64 image data")
		}

		if mediaType == "" {
			mediaType = llm.DetectImageMediaType(data)
		}
		if mediaType == "" {
			mediaType = "image/png"
		}

		images = append(images, llm.NewGeneratedImage(data, mediaType, img.RevisedPrompt))
	}

	usage := llm.NewImageGenerationUsage(parsed.Usage.PromptTokens, parsed.Usage.CompletionTokens, parsed.Usage.TotalTokens)

	return llm.NewImageGenerationResponse(images, usage), nil
}

// NewImageGenerationClient construit le client. baseURL vide vaut le service
// public ; sinon elle doit pointer la racine de l'API (".../api/v1"), le
// suffixe "/images" est ajouté ici.
func NewImageGenerationClient(httpClient *http.Client, baseURL, apiKey, model string) *ImageGenerationClient {
	endpoint := imageGenerationEndpoint
	if baseURL != "" {
		endpoint = strings.TrimSuffix(baseURL, "/") + "/images"
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &ImageGenerationClient{httpClient: httpClient, endpoint: endpoint, apiKey: apiKey, model: model}
}

var _ llm.ImageGenerationClient = &ImageGenerationClient{}

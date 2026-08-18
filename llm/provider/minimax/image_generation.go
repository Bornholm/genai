package minimax

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

// ImageGenerationClient generates images through the MiniMax API
// (POST {base}/image_generation, réponse data.image_base64[]).
type ImageGenerationClient struct {
	httpClient *http.Client
	endpoint   string
	apiKey     string
	model      string
}

type imageGenerationRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	// ResponseFormat vaut toujours "base64" : l'appelant reçoit des octets,
	// jamais une URL éphémère à récupérer dans la foulée.
	ResponseFormat string `json:"response_format"`
	AspectRatio    string `json:"aspect_ratio,omitempty"`
	N              int    `json:"n,omitempty"`
}

type imageGenerationResponse struct {
	Data struct {
		ImageBase64 []string `json:"image_base64"`
	} `json:"data"`
	// BaseResp est le bloc de statut MiniMax : status_code 0 = succès. Les
	// erreurs applicatives (clé invalide, prompt refusé) arrivent souvent
	// avec un HTTP 200 et un status_code non nul.
	BaseResp struct {
		StatusCode int    `json:"status_code"`
		StatusMsg  string `json:"status_msg"`
	} `json:"base_resp"`
}

// ImageGeneration implements [llm.ImageGenerationClient].
func (c *ImageGenerationClient) ImageGeneration(ctx context.Context, prompt string, funcs ...llm.ImageGenerationOptionFunc) (llm.ImageGenerationResponse, error) {
	opts := llm.NewImageGenerationOptions(funcs...)

	payload := imageGenerationRequest{
		Model:          c.model,
		Prompt:         prompt,
		ResponseFormat: "base64",
		AspectRatio:    opts.AspectRatio,
		N:              opts.Count,
	}
	// Size est le dialecte OpenAI ("1024x1024") : traduit vers le ratio
	// équivalent quand aucun ratio n'est demandé explicitement.
	if payload.AspectRatio == "" && opts.Size != "" {
		payload.AspectRatio = sizeToAspectRatio(opts.Size)
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

	if parsed.BaseResp.StatusCode != 0 {
		return nil, errors.Errorf("minimax error %d: %s", parsed.BaseResp.StatusCode, parsed.BaseResp.StatusMsg)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errors.Errorf("unexpected HTTP status %d", resp.StatusCode)
	}
	if len(parsed.Data.ImageBase64) == 0 {
		return nil, errors.New("no image in response")
	}

	images := make([]llm.GeneratedImage, 0, len(parsed.Data.ImageBase64))
	for _, b64 := range parsed.Data.ImageBase64 {
		data, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			return nil, errors.Wrap(err, "could not decode base64 image data")
		}

		mediaType := llm.DetectImageMediaType(data)
		if mediaType == "" {
			mediaType = "image/jpeg"
		}

		images = append(images, llm.NewGeneratedImage(data, mediaType, ""))
	}

	return llm.NewImageGenerationResponse(images, nil), nil
}

// sizeToAspectRatio traduit une taille OpenAI vers le ratio MiniMax le plus
// proche. Une taille inconnue ne produit rien : le défaut du modèle vaut
// mieux qu'un ratio deviné de travers.
func sizeToAspectRatio(size string) string {
	switch size {
	case "256x256", "512x512", "1024x1024":
		return "1:1"
	case "1792x1024", "1536x1024":
		return "16:9"
	case "1024x1792", "1024x1536":
		return "9:16"
	}
	return ""
}

// NewImageGenerationClient construit le client. baseURL vide vaut le service
// public ; sinon elle doit pointer la racine versionnée de l'API
// (".../v1"), le suffixe "/image_generation" est ajouté ici.
func NewImageGenerationClient(httpClient *http.Client, baseURL, apiKey, model string) *ImageGenerationClient {
	if baseURL == "" {
		baseURL = "https://api.minimax.io/v1"
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &ImageGenerationClient{
		httpClient: httpClient,
		endpoint:   strings.TrimSuffix(baseURL, "/") + "/image_generation",
		apiKey:     apiKey,
		model:      model,
	}
}

var _ llm.ImageGenerationClient = &ImageGenerationClient{}

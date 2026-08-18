package openai

import (
	"context"
	"encoding/base64"

	"github.com/pkg/errors"

	"github.com/bornholm/genai/llm"

	openaisdk "github.com/openai/openai-go"
)

// ImageGenerationClient generates images through the OpenAI images API
// (/v1/images/generations).
type ImageGenerationClient struct {
	client openaisdk.Client
	model  string
}

// ImageGeneration implements [llm.ImageGenerationClient].
func (c *ImageGenerationClient) ImageGeneration(ctx context.Context, prompt string, funcs ...llm.ImageGenerationOptionFunc) (llm.ImageGenerationResponse, error) {
	opts := llm.NewImageGenerationOptions(funcs...)

	params := openaisdk.ImageGenerateParams{
		Prompt: prompt,
		Model:  c.model,
		// b64_json plutôt que l'URL par défaut : l'appelant reçoit des
		// octets, jamais une URL à durée de vie limitée (60 minutes côté
		// OpenAI) qu'il faudrait aller chercher dans la foulée.
		ResponseFormat: openaisdk.ImageGenerateParamsResponseFormatB64JSON,
	}

	if opts.Count > 0 {
		params.N = openaisdk.Int(int64(opts.Count))
	}
	if opts.Size != "" {
		params.Size = openaisdk.ImageGenerateParamsSize(opts.Size)
	}
	if opts.Quality != "" {
		params.Quality = openaisdk.ImageGenerateParamsQuality(opts.Quality)
	}
	// AspectRatio est le dialecte MiniMax : ce provider ne le traduit pas,
	// les tailles OpenAI étant une liste fermée dépendante du modèle.

	resp, err := c.client.Images.Generate(ctx, params)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	images := make([]llm.GeneratedImage, 0, len(resp.Data))
	for _, img := range resp.Data {
		data, err := base64.StdEncoding.DecodeString(img.B64JSON)
		if err != nil {
			return nil, errors.Wrap(err, "could not decode base64 image data")
		}

		mediaType := llm.DetectImageMediaType(data)
		if mediaType == "" {
			mediaType = "image/png"
		}

		images = append(images, llm.NewGeneratedImage(data, mediaType, img.RevisedPrompt))
	}

	return llm.NewImageGenerationResponse(images, nil), nil
}

func NewImageGenerationClient(client openaisdk.Client, model string) *ImageGenerationClient {
	return &ImageGenerationClient{client: client, model: model}
}

var _ llm.ImageGenerationClient = &ImageGenerationClient{}

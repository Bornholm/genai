package llm

import "context"

// ImageGenerationClient generates images from a textual prompt.
//
// It is deliberately a separate interface rather than a member of [Client]:
// adding a method to [Client] breaks every existing implementation, as the
// introduction of [TranscriptionClient] did. Callers discover the capability
// with a type assertion:
//
//	if generator, ok := client.(llm.ImageGenerationClient); ok {
//	    // ...
//	}
type ImageGenerationClient interface {
	ImageGeneration(ctx context.Context, prompt string, funcs ...ImageGenerationOptionFunc) (ImageGenerationResponse, error)
}

// ImageGenerationOptions gathers the options shared by image providers.
//
// Providers ignore what they do not support rather than failing: the same
// call must remain usable across providers, which is the whole point of the
// interface. Size and AspectRatio express the same intent in the two
// dialects found in the wild — providers map one to the other when they can.
type ImageGenerationOptions struct {
	// Count is the number of images requested. Zero means one.
	Count int
	// Size is a pixel size such as "1024x1024" (OpenAI dialect).
	Size string
	// AspectRatio is a ratio such as "16:9" (MiniMax dialect).
	AspectRatio string
	// Quality is a provider-specific quality level, e.g. "standard" or "hd".
	Quality string
}

func NewImageGenerationOptions(funcs ...ImageGenerationOptionFunc) *ImageGenerationOptions {
	opts := &ImageGenerationOptions{}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

type ImageGenerationOptionFunc func(opts *ImageGenerationOptions)

func WithImageCount(count int) ImageGenerationOptionFunc {
	return func(opts *ImageGenerationOptions) {
		opts.Count = count
	}
}

func WithImageSize(size string) ImageGenerationOptionFunc {
	return func(opts *ImageGenerationOptions) {
		opts.Size = size
	}
}

func WithImageAspectRatio(ratio string) ImageGenerationOptionFunc {
	return func(opts *ImageGenerationOptions) {
		opts.AspectRatio = ratio
	}
}

func WithImageQuality(quality string) ImageGenerationOptionFunc {
	return func(opts *ImageGenerationOptions) {
		opts.Quality = quality
	}
}

// ImageGenerationResponse carries the generated images.
type ImageGenerationResponse interface {
	Images() []GeneratedImage
	// Usage may be nil when the provider reports no metrics.
	Usage() ImageGenerationUsage
}

// GeneratedImage is a single generated image, already decoded.
//
// Providers return base64 or a URL depending on the endpoint; implementations
// normalise to raw bytes so callers never deal with either. A provider that
// only returns a URL fetches it before returning.
type GeneratedImage interface {
	// Data returns the raw image bytes.
	Data() []byte
	// MediaType is the IANA type of the bytes, e.g. "image/png".
	MediaType() string
	// RevisedPrompt is the prompt as rewritten by the model, empty when the
	// provider does not report one.
	RevisedPrompt() string
}

type ImageGenerationUsage interface {
	InputTokens() int64
	OutputTokens() int64
	TotalTokens() int64
}

type BaseGeneratedImage struct {
	data          []byte
	mediaType     string
	revisedPrompt string
}

// Data implements GeneratedImage.
func (i *BaseGeneratedImage) Data() []byte { return i.data }

// MediaType implements GeneratedImage.
func (i *BaseGeneratedImage) MediaType() string { return i.mediaType }

// RevisedPrompt implements GeneratedImage.
func (i *BaseGeneratedImage) RevisedPrompt() string { return i.revisedPrompt }

func NewGeneratedImage(data []byte, mediaType, revisedPrompt string) *BaseGeneratedImage {
	return &BaseGeneratedImage{data: data, mediaType: mediaType, revisedPrompt: revisedPrompt}
}

var _ GeneratedImage = &BaseGeneratedImage{}

type BaseImageGenerationUsage struct {
	inputTokens  int64
	outputTokens int64
	totalTokens  int64
	cost         *float64
	costCurrency string
}

// InputTokens implements ImageGenerationUsage.
func (u *BaseImageGenerationUsage) InputTokens() int64 { return u.inputTokens }

// OutputTokens implements ImageGenerationUsage.
func (u *BaseImageGenerationUsage) OutputTokens() int64 { return u.outputTokens }

// TotalTokens implements ImageGenerationUsage.
func (u *BaseImageGenerationUsage) TotalTokens() int64 { return u.totalTokens }

func NewImageGenerationUsage(inputTokens, outputTokens, totalTokens int64) *BaseImageGenerationUsage {
	return &BaseImageGenerationUsage{inputTokens: inputTokens, outputTokens: outputTokens, totalTokens: totalTokens}
}

// NewImageGenerationUsageWithCost reports what the generation actually
// cost. Images are billed per output, not per token: an estimate built
// from token counts is far off, and gateways state the real price.
func NewImageGenerationUsageWithCost(inputTokens, outputTokens, totalTokens int64, cost float64, currency string) *BaseImageGenerationUsage {
	return &BaseImageGenerationUsage{
		inputTokens:  inputTokens,
		outputTokens: outputTokens,
		totalTokens:  totalTokens,
		cost:         &cost,
		costCurrency: currency,
	}
}

// Cost implements CostReportingUsage.
func (u *BaseImageGenerationUsage) Cost() (amount float64, currency string, ok bool) {
	if u.cost == nil {
		return 0, "", false
	}
	return *u.cost, u.costCurrency, true
}

var (
	_ ImageGenerationUsage = &BaseImageGenerationUsage{}
	_ CostReportingUsage   = &BaseImageGenerationUsage{}
)

type BaseImageGenerationResponse struct {
	images []GeneratedImage
	usage  ImageGenerationUsage
}

// Images implements ImageGenerationResponse.
func (r *BaseImageGenerationResponse) Images() []GeneratedImage { return r.images }

// Usage implements ImageGenerationResponse.
func (r *BaseImageGenerationResponse) Usage() ImageGenerationUsage { return r.usage }

func NewImageGenerationResponse(images []GeneratedImage, usage ImageGenerationUsage) *BaseImageGenerationResponse {
	return &BaseImageGenerationResponse{images: images, usage: usage}
}

var _ ImageGenerationResponse = &BaseImageGenerationResponse{}

// DetectImageMediaType returns the IANA media type of the given bytes, or ""
// when unrecognised. Providers that omit the type still yield something
// usable to the caller — a data: URI or a file needs one.
func DetectImageMediaType(data []byte) string {
	switch {
	case len(data) >= 8 && string(data[:8]) == "\x89PNG\r\n\x1a\n":
		return "image/png"
	case len(data) >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF:
		return "image/jpeg"
	case len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP":
		return "image/webp"
	case len(data) >= 6 && (string(data[:6]) == "GIF87a" || string(data[:6]) == "GIF89a"):
		return "image/gif"
	case len(data) >= 5 && string(data[:5]) == "<?xml", len(data) >= 4 && string(data[:4]) == "<svg":
		return "image/svg+xml"
	}
	return ""
}

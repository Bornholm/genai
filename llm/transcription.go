package llm

import (
	"bytes"
	"context"
)

// AudioFormat identifie le format d'un fichier audio soumis à la transcription.
type AudioFormat string

const (
	AudioFormatMP3  AudioFormat = "mp3"
	AudioFormatWAV  AudioFormat = "wav"
	AudioFormatFLAC AudioFormat = "flac"
	AudioFormatOGG  AudioFormat = "ogg"
	AudioFormatM4A  AudioFormat = "m4a"
	AudioFormatWEBM AudioFormat = "webm"
	AudioFormatAAC  AudioFormat = "aac"
)

type TranscriptionClient interface {
	Transcription(ctx context.Context, audio []byte, funcs ...TranscriptionOptionFunc) (TranscriptionResponse, error)
}

type TranscriptionOptions struct {
	// Format indique le format de l'audio. S'il est vide, les providers
	// tentent de le détecter via DetectAudioFormat.
	Format AudioFormat
	// Language est un code ISO-639-1 (ex. "fr") améliorant précision et latence.
	Language string
	// Prompt guide le style du modèle (non supporté par tous les providers).
	Prompt string
	// Temperature contrôle l'échantillonnage, entre 0 et 1.
	Temperature *float64
}

func NewTranscriptionOptions(funcs ...TranscriptionOptionFunc) *TranscriptionOptions {
	opts := &TranscriptionOptions{}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

type TranscriptionOptionFunc func(opts *TranscriptionOptions)

func WithAudioFormat(format AudioFormat) TranscriptionOptionFunc {
	return func(opts *TranscriptionOptions) {
		opts.Format = format
	}
}

func WithTranscriptionLanguage(language string) TranscriptionOptionFunc {
	return func(opts *TranscriptionOptions) {
		opts.Language = language
	}
}

func WithTranscriptionPrompt(prompt string) TranscriptionOptionFunc {
	return func(opts *TranscriptionOptions) {
		opts.Prompt = prompt
	}
}

func WithTranscriptionTemperature(temperature float64) TranscriptionOptionFunc {
	return func(opts *TranscriptionOptions) {
		opts.Temperature = &temperature
	}
}

type TranscriptionResponse interface {
	Text() string
	// Language est le code de la langue détectée, vide si le provider ne le fournit pas.
	Language() string
	// Usage peut être nil si le provider ne fournit pas de métriques.
	Usage() TranscriptionUsage
}

type TranscriptionUsage interface {
	InputTokens() int64
	OutputTokens() int64
	TotalTokens() int64
	// AudioSeconds est la durée de l'audio facturée, 0 si inconnue.
	AudioSeconds() float64
}

type BaseTranscriptionUsage struct {
	inputTokens  int64
	outputTokens int64
	totalTokens  int64
	audioSeconds float64
	cost         *float64
	costCurrency string
}

// InputTokens implements TranscriptionUsage.
func (u *BaseTranscriptionUsage) InputTokens() int64 {
	return u.inputTokens
}

// OutputTokens implements TranscriptionUsage.
func (u *BaseTranscriptionUsage) OutputTokens() int64 {
	return u.outputTokens
}

// TotalTokens implements TranscriptionUsage.
func (u *BaseTranscriptionUsage) TotalTokens() int64 {
	return u.totalTokens
}

// AudioSeconds implements TranscriptionUsage.
func (u *BaseTranscriptionUsage) AudioSeconds() float64 {
	return u.audioSeconds
}

func NewTranscriptionUsage(inputTokens, outputTokens, totalTokens int64, audioSeconds float64) *BaseTranscriptionUsage {
	return &BaseTranscriptionUsage{
		inputTokens:  inputTokens,
		outputTokens: outputTokens,
		totalTokens:  totalTokens,
		audioSeconds: audioSeconds,
	}
}

// NewTranscriptionUsageWithCost reports what the transcription actually
// cost, as gateways such as OpenRouter do. Use CostReportingUsage to read
// it back: billing from seconds or tokens is a guess, this is not.
func NewTranscriptionUsageWithCost(inputTokens, outputTokens, totalTokens int64, audioSeconds float64, cost float64, currency string) *BaseTranscriptionUsage {
	return &BaseTranscriptionUsage{
		inputTokens:  inputTokens,
		outputTokens: outputTokens,
		totalTokens:  totalTokens,
		audioSeconds: audioSeconds,
		cost:         &cost,
		costCurrency: currency,
	}
}

// Cost implements CostReportingUsage.
func (u *BaseTranscriptionUsage) Cost() (amount float64, currency string, ok bool) {
	if u.cost == nil {
		return 0, "", false
	}
	return *u.cost, u.costCurrency, true
}

var (
	_ TranscriptionUsage = &BaseTranscriptionUsage{}
	_ CostReportingUsage = &BaseTranscriptionUsage{}
)

type BaseTranscriptionResponse struct {
	text     string
	language string
	usage    TranscriptionUsage
}

// Text implements TranscriptionResponse.
func (r *BaseTranscriptionResponse) Text() string {
	return r.text
}

// Language implements TranscriptionResponse.
func (r *BaseTranscriptionResponse) Language() string {
	return r.language
}

// Usage implements TranscriptionResponse.
func (r *BaseTranscriptionResponse) Usage() TranscriptionUsage {
	return r.usage
}

func NewTranscriptionResponse(text, language string, usage TranscriptionUsage) *BaseTranscriptionResponse {
	return &BaseTranscriptionResponse{
		text:     text,
		language: language,
		usage:    usage,
	}
}

var _ TranscriptionResponse = &BaseTranscriptionResponse{}

// DetectAudioFormat détecte le format d'un flux audio à partir de ses premiers
// octets (magic numbers). Retourne "" si le format n'est pas reconnu.
func DetectAudioFormat(audio []byte) AudioFormat {
	if len(audio) < 12 {
		return ""
	}

	switch {
	case bytes.HasPrefix(audio, []byte("ID3")),
		len(audio) > 1 && audio[0] == 0xFF && (audio[1] == 0xFB || audio[1] == 0xFA || audio[1] == 0xF3 || audio[1] == 0xF2):
		return AudioFormatMP3
	case bytes.HasPrefix(audio, []byte("RIFF")) && bytes.Equal(audio[8:12], []byte("WAVE")):
		return AudioFormatWAV
	case bytes.HasPrefix(audio, []byte("fLaC")):
		return AudioFormatFLAC
	case bytes.HasPrefix(audio, []byte("OggS")):
		return AudioFormatOGG
	case bytes.Equal(audio[4:8], []byte("ftyp")):
		return AudioFormatM4A
	case bytes.HasPrefix(audio, []byte{0x1A, 0x45, 0xDF, 0xA3}):
		return AudioFormatWEBM
	case audio[0] == 0xFF && (audio[1] == 0xF1 || audio[1] == 0xF9):
		return AudioFormatAAC
	}

	return ""
}

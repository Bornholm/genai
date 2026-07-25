package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/pkg/errors"
)

type TranscriptionClient struct {
	client openai.Client
	model  string
}

// Transcription implements llm.TranscriptionClient.
func (c *TranscriptionClient) Transcription(ctx context.Context, audio []byte, funcs ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	if c.model == "" {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	opts := llm.NewTranscriptionOptions(funcs...)

	format := opts.Format
	if format == "" {
		format = llm.DetectAudioFormat(audio)
	}
	if format == "" {
		format = llm.AudioFormatMP3
	}

	params := openai.AudioTranscriptionNewParams{
		File:  openai.File(bytes.NewReader(audio), "audio."+string(format), audioMimeType(format)),
		Model: openai.AudioModel(c.model),
	}

	if opts.Language != "" {
		params.Language = openai.String(opts.Language)
	}

	if opts.Prompt != "" {
		params.Prompt = openai.String(opts.Prompt)
	}

	if opts.Temperature != nil {
		params.Temperature = openai.Float(*opts.Temperature)
	}

	var httpRes *http.Response

	slog.DebugContext(ctx, "starting transcription")
	before := time.Now()
	res, err := c.client.Audio.Transcriptions.New(ctx, params, option.WithResponseInto(&httpRes))
	slog.DebugContext(ctx, "transcription completed", slog.Duration("duration", time.Since(before)))

	if err != nil {
		if httpRes != nil {
			body, _ := io.ReadAll(httpRes.Body)
			return nil, errors.WithStack(llm.RateLimitError(httpRes.StatusCode, string(body)))
		}

		return nil, errors.WithStack(err)
	}

	language, usage := parseTranscriptionMetadata([]byte(res.RawJSON()))

	return llm.NewTranscriptionResponse(res.Text, language, usage), nil
}

// rawTranscriptionMetadata couvre les champs additionnels retournés par les
// différentes APIs compatibles OpenAI (OpenAI, Mistral, OpenRouter), que le SDK
// ne parse pas dans openai.Transcription.
type rawTranscriptionMetadata struct {
	Language string  `json:"language"`
	Duration float64 `json:"duration"`
	Usage    *struct {
		InputTokens        int64   `json:"input_tokens"`
		OutputTokens       int64   `json:"output_tokens"`
		TotalTokens        int64   `json:"total_tokens"`
		PromptTokens       int64   `json:"prompt_tokens"`
		CompletionTokens   int64   `json:"completion_tokens"`
		PromptAudioSeconds float64 `json:"prompt_audio_seconds"`
		Seconds            float64 `json:"seconds"`
	} `json:"usage"`
}

func parseTranscriptionMetadata(raw []byte) (string, llm.TranscriptionUsage) {
	var meta rawTranscriptionMetadata
	if err := json.Unmarshal(raw, &meta); err != nil {
		return "", nil
	}

	if meta.Usage == nil {
		if meta.Duration > 0 {
			return meta.Language, llm.NewTranscriptionUsage(0, 0, 0, meta.Duration)
		}
		return meta.Language, nil
	}

	inputTokens := meta.Usage.InputTokens
	if inputTokens == 0 {
		inputTokens = meta.Usage.PromptTokens
	}

	outputTokens := meta.Usage.OutputTokens
	if outputTokens == 0 {
		outputTokens = meta.Usage.CompletionTokens
	}

	audioSeconds := meta.Usage.Seconds
	if audioSeconds == 0 {
		audioSeconds = meta.Usage.PromptAudioSeconds
	}
	if audioSeconds == 0 {
		audioSeconds = meta.Duration
	}

	return meta.Language, llm.NewTranscriptionUsage(inputTokens, outputTokens, meta.Usage.TotalTokens, audioSeconds)
}

func audioMimeType(format llm.AudioFormat) string {
	switch format {
	case llm.AudioFormatMP3:
		return "audio/mpeg"
	case llm.AudioFormatWAV:
		return "audio/wav"
	case llm.AudioFormatFLAC:
		return "audio/flac"
	case llm.AudioFormatOGG:
		return "audio/ogg"
	case llm.AudioFormatM4A:
		return "audio/mp4"
	case llm.AudioFormatWEBM:
		return "audio/webm"
	case llm.AudioFormatAAC:
		return "audio/aac"
	default:
		return "application/octet-stream"
	}
}

func NewTranscriptionClient(client openai.Client, model string) *TranscriptionClient {
	return &TranscriptionClient{
		client: client,
		model:  model,
	}
}

var _ llm.TranscriptionClient = &TranscriptionClient{}

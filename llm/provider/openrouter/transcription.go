package openrouter

import (
	"context"
	"log/slog"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"github.com/revrost/go-openrouter"
)

type TranscriptionClient struct {
	client *openrouter.Client
	model  string
}

// Transcription implements [llm.TranscriptionClient].
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

	req := openrouter.TranscriptionRequest{
		Model:      c.model,
		InputAudio: openrouter.NewTranscriptionInputAudio(audio, openrouter.AudioFormat(format)),
	}

	if opts.Language != "" {
		req.Language = opts.Language
	}

	if opts.Temperature != nil {
		req.Temperature = opts.Temperature
	}

	slog.DebugContext(ctx, "starting transcription")
	before := time.Now()
	res, err := c.client.CreateTranscription(ctx, req)
	slog.DebugContext(ctx, "transcription completed", slog.Duration("duration", time.Since(before)))

	if err != nil {
		var reqErr *openrouter.RequestError
		if errors.As(err, &reqErr) {
			return nil, errors.WithStack(llm.RateLimitError(reqErr.HTTPStatusCode, reqErr.Error()))
		}

		return nil, errors.WithStack(err)
	}

	var usage llm.TranscriptionUsage
	if res.Usage != nil {
		usage = llm.NewTranscriptionUsage(
			int64(res.Usage.InputTokens),
			int64(res.Usage.OutputTokens),
			int64(res.Usage.TotalTokens),
			res.Usage.Seconds,
		)
	}

	return llm.NewTranscriptionResponse(res.Text, "", usage), nil
}

func NewTranscriptionClient(client *openrouter.Client, model string) *TranscriptionClient {
	return &TranscriptionClient{
		client: client,
		model:  model,
	}
}

var _ llm.TranscriptionClient = &TranscriptionClient{}

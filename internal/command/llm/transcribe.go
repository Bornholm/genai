package llm

import (
	"log/slog"
	"os"
	"time"

	"github.com/bornholm/genai/internal/command/common"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"
)

func Transcribe() *cli.Command {
	return &cli.Command{
		Name:  "transcribe",
		Usage: "Transcribe an audio file (speech-to-text)",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:      "file",
				Aliases:   []string{"f"},
				Usage:     "Audio file to transcribe",
				EnvVars:   []string{"GENAI_TRANSCRIBE_FILE"},
				Required:  true,
				TakesFile: true,
			},
			&cli.StringFlag{
				Name:    "language",
				Aliases: []string{"l"},
				Usage:   "Audio language as an ISO-639-1 code (e.g. 'fr'), auto-detected if empty",
				EnvVars: []string{"GENAI_TRANSCRIBE_LANGUAGE"},
			},
			&cli.StringFlag{
				Name:    "audio-format",
				Usage:   "Audio format (mp3, wav, flac, ogg, m4a, webm, aac), auto-detected if empty",
				EnvVars: []string{"GENAI_TRANSCRIBE_AUDIO_FORMAT"},
			},
			&cli.StringFlag{
				Name:    "prompt",
				Usage:   "Optional text to guide the model's style (not supported by all providers)",
				EnvVars: []string{"GENAI_TRANSCRIBE_PROMPT"},
			},
			&cli.Float64Flag{
				Name:    "temperature",
				Usage:   "Sampling temperature (0.0 to 1.0)",
				EnvVars: []string{"GENAI_TRANSCRIBE_TEMPERATURE"},
			},
			&cli.StringFlag{
				Name:      "env-file",
				Usage:     "Environment file path",
				EnvVars:   []string{"GENAI_LLM_ENV_FILE"},
				Value:     ".env",
				TakesFile: true,
			},
			&cli.StringFlag{
				Name:    "env-prefix",
				Usage:   "Environment llm variables prefix",
				EnvVars: []string{"GENAI_LLM_ENV_PREFIX"},
				Value:   "GENAI_",
			},
			&cli.StringFlag{
				Name:    "output",
				Aliases: []string{"o"},
				Usage:   "Output file path (default: stdout)",
				EnvVars: []string{"GENAI_OUTPUT"},
			},
		},
		Action: func(cliCtx *cli.Context) error {
			ctx := cliCtx.Context

			envPrefix := cliCtx.String("env-prefix")
			envFile := cliCtx.String("env-file")

			client, err := common.NewResilientClient(ctx, envPrefix, envFile, nil)
			if err != nil {
				return errors.Wrap(err, "failed to create llm client")
			}

			audio, err := os.ReadFile(cliCtx.String("file"))
			if err != nil {
				return errors.Wrap(err, "failed to read audio file")
			}

			opts := []llm.TranscriptionOptionFunc{}

			if language := cliCtx.String("language"); language != "" {
				opts = append(opts, llm.WithTranscriptionLanguage(language))
			}

			if format := cliCtx.String("audio-format"); format != "" {
				opts = append(opts, llm.WithAudioFormat(llm.AudioFormat(format)))
			}

			if prompt := cliCtx.String("prompt"); prompt != "" {
				opts = append(opts, llm.WithTranscriptionPrompt(prompt))
			}

			if cliCtx.IsSet("temperature") {
				opts = append(opts, llm.WithTranscriptionTemperature(cliCtx.Float64("temperature")))
			}

			before := time.Now()
			response, err := client.Transcription(ctx, audio, opts...)
			if err != nil {
				return errors.Wrap(err, "failed to transcribe audio")
			}

			if err := common.WriteToOutput(cliCtx, "output", response.Text(), false); err != nil {
				return errors.Wrap(err, "failed to write to output")
			}

			logAttrs := []any{
				slog.Duration("duration", time.Since(before)),
			}
			if response.Language() != "" {
				logAttrs = append(logAttrs, slog.String("language", response.Language()))
			}
			if usage := response.Usage(); usage != nil {
				logAttrs = append(logAttrs,
					slog.Int64("input_tokens", usage.InputTokens()),
					slog.Int64("output_tokens", usage.OutputTokens()),
					slog.Int64("total_tokens", usage.TotalTokens()),
					slog.Float64("audio_seconds", usage.AudioSeconds()),
				)
			}
			slog.DebugContext(ctx, "Transcription completed", logAttrs...)

			return nil
		},
	}
}

package main

import (
	"context"
	"flag"
	"log"
	"os"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"

	_ "github.com/bornholm/genai/llm/provider/all"
	"github.com/bornholm/genai/llm/provider/env"
)

var (
	envFile   string = ".env"
	audioFile string = "hello_world.mp3"
	language  string = ""
)

func init() {
	flag.StringVar(&envFile, "env-file", envFile, "client configuration environment file")
	flag.StringVar(&audioFile, "audio", audioFile, "audio file to transcribe")
	flag.StringVar(&language, "language", language, "audio language as an ISO-639-1 code (e.g. 'fr'), auto-detected if empty")
}

func main() {
	flag.Parse()
	ctx := context.Background()

	client, err := provider.Create(ctx, env.With("GENAI_", envFile))
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	audio, err := os.ReadFile(audioFile)
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	opts := []llm.TranscriptionOptionFunc{}
	if language != "" {
		opts = append(opts, llm.WithTranscriptionLanguage(language))
	}

	res, err := client.Transcription(ctx, audio, opts...)
	if err != nil {
		log.Fatalf("[FATAL] %s", err)
	}

	log.Printf("[TRANSCRIPT] %s", res.Text())

	if res.Language() != "" {
		log.Printf("[INFO] Detected language: %s", res.Language())
	}

	if usage := res.Usage(); usage != nil {
		log.Printf("[INFO] Usage: input tokens=%d, output tokens=%d, total tokens=%d, audio seconds=%.1f",
			usage.InputTokens(), usage.OutputTokens(), usage.TotalTokens(), usage.AudioSeconds())
	}
}

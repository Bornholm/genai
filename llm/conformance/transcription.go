package conformance

import (
	"context"
	_ "embed"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

// helloWorldAudio est un court extrait audio synthétique (espeak-ng) disant "hello world".
//
//go:embed testdata/hello_world.mp3
var helloWorldAudio []byte

func testTranscription(t *testing.T, client any) {
	t.Helper()

	sttClient, ok := client.(llm.TranscriptionClient)
	if !ok {
		t.Skip("client does not implement TranscriptionClient")
	}

	ctx := context.Background()

	t.Run("HelloWorld", func(t *testing.T) {
		res, err := sttClient.Transcription(ctx, helloWorldAudio,
			llm.WithTranscriptionLanguage("en"),
		)
		if err != nil {
			t.Fatalf("Transcription error: %v", err)
		}

		if res.Text() == "" {
			t.Fatal("expected non-empty transcription text")
		}

		if !strings.Contains(strings.ToLower(res.Text()), "hello") {
			t.Errorf("expected transcription to contain 'hello', got %q", res.Text())
		}
	})

	t.Run("ExplicitFormat", func(t *testing.T) {
		res, err := sttClient.Transcription(ctx, helloWorldAudio,
			llm.WithAudioFormat(llm.AudioFormatMP3),
		)
		if err != nil {
			t.Fatalf("Transcription error: %v", err)
		}

		if res.Text() == "" {
			t.Fatal("expected non-empty transcription text")
		}
	})
}

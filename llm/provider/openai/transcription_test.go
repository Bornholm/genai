package openai

import (
	"testing"

	"github.com/bornholm/genai/llm"
)

func TestParseTranscriptionMetadata_OpenAI(t *testing.T) {
	raw := []byte(`{"text":"hello","usage":{"type":"tokens","input_tokens":14,"output_tokens":45,"total_tokens":59}}`)

	language, usage := parseTranscriptionMetadata(raw)

	if language != "" {
		t.Errorf("expected empty language, got %q", language)
	}
	if usage == nil {
		t.Fatal("expected non-nil usage")
	}
	if usage.InputTokens() != 14 {
		t.Errorf("expected 14 input tokens, got %d", usage.InputTokens())
	}
	if usage.OutputTokens() != 45 {
		t.Errorf("expected 45 output tokens, got %d", usage.OutputTokens())
	}
	if usage.TotalTokens() != 59 {
		t.Errorf("expected 59 total tokens, got %d", usage.TotalTokens())
	}
}

func TestParseTranscriptionMetadata_Mistral(t *testing.T) {
	raw := []byte(`{"model":"voxtral-mini-2507","text":"bonjour","language":"fr","segments":[],"usage":{"prompt_audio_seconds":203,"prompt_tokens":4,"total_tokens":3264,"completion_tokens":635}}`)

	language, usage := parseTranscriptionMetadata(raw)

	if language != "fr" {
		t.Errorf("expected language 'fr', got %q", language)
	}
	if usage == nil {
		t.Fatal("expected non-nil usage")
	}
	if usage.InputTokens() != 4 {
		t.Errorf("expected 4 input tokens, got %d", usage.InputTokens())
	}
	if usage.OutputTokens() != 635 {
		t.Errorf("expected 635 output tokens, got %d", usage.OutputTokens())
	}
	if usage.TotalTokens() != 3264 {
		t.Errorf("expected 3264 total tokens, got %d", usage.TotalTokens())
	}
	if usage.AudioSeconds() != 203 {
		t.Errorf("expected 203 audio seconds, got %f", usage.AudioSeconds())
	}
}

func TestParseTranscriptionMetadata_VerboseJSON(t *testing.T) {
	raw := []byte(`{"text":"hello","language":"english","duration":8.47}`)

	language, usage := parseTranscriptionMetadata(raw)

	if language != "english" {
		t.Errorf("expected language 'english', got %q", language)
	}
	if usage == nil {
		t.Fatal("expected non-nil usage")
	}
	if usage.AudioSeconds() != 8.47 {
		t.Errorf("expected 8.47 audio seconds, got %f", usage.AudioSeconds())
	}
}

func TestParseTranscriptionMetadata_NoUsage(t *testing.T) {
	raw := []byte(`{"text":"hello"}`)

	language, usage := parseTranscriptionMetadata(raw)

	if language != "" {
		t.Errorf("expected empty language, got %q", language)
	}
	if usage != nil {
		t.Errorf("expected nil usage, got %v", usage)
	}
}

func TestAudioMimeType(t *testing.T) {
	testCases := []struct {
		format   llm.AudioFormat
		expected string
	}{
		{llm.AudioFormatMP3, "audio/mpeg"},
		{llm.AudioFormatWAV, "audio/wav"},
		{llm.AudioFormatFLAC, "audio/flac"},
		{llm.AudioFormatOGG, "audio/ogg"},
		{llm.AudioFormatM4A, "audio/mp4"},
		{llm.AudioFormatWEBM, "audio/webm"},
		{llm.AudioFormatAAC, "audio/aac"},
		{llm.AudioFormat("unknown"), "application/octet-stream"},
	}

	for _, tc := range testCases {
		if got := audioMimeType(tc.format); got != tc.expected {
			t.Errorf("expected mime type %q for format %q, got %q", tc.expected, tc.format, got)
		}
	}
}

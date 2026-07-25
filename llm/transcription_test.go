package llm

import (
	"testing"
)

func TestNewTranscriptionOptions(t *testing.T) {
	opts := NewTranscriptionOptions(
		WithAudioFormat(AudioFormatWAV),
		WithTranscriptionLanguage("fr"),
		WithTranscriptionPrompt("technical vocabulary"),
		WithTranscriptionTemperature(0.2),
	)

	if opts.Format != AudioFormatWAV {
		t.Errorf("expected format 'wav', got %q", opts.Format)
	}
	if opts.Language != "fr" {
		t.Errorf("expected language 'fr', got %q", opts.Language)
	}
	if opts.Prompt != "technical vocabulary" {
		t.Errorf("expected prompt 'technical vocabulary', got %q", opts.Prompt)
	}
	if opts.Temperature == nil || *opts.Temperature != 0.2 {
		t.Errorf("expected temperature 0.2, got %v", opts.Temperature)
	}
}

func TestNewTranscriptionOptions_Defaults(t *testing.T) {
	opts := NewTranscriptionOptions()

	if opts.Format != "" {
		t.Errorf("expected empty format, got %q", opts.Format)
	}
	if opts.Temperature != nil {
		t.Errorf("expected nil temperature, got %v", opts.Temperature)
	}
}

func TestDetectAudioFormat(t *testing.T) {
	pad := func(prefix []byte) []byte {
		data := make([]byte, 16)
		copy(data, prefix)
		return data
	}

	wav := pad([]byte("RIFF"))
	copy(wav[8:], []byte("WAVE"))

	m4a := pad(nil)
	copy(m4a[4:], []byte("ftyp"))

	testCases := []struct {
		name     string
		data     []byte
		expected AudioFormat
	}{
		{"mp3 with ID3 tag", pad([]byte("ID3")), AudioFormatMP3},
		{"mp3 frame sync", pad([]byte{0xFF, 0xFB}), AudioFormatMP3},
		{"wav", wav, AudioFormatWAV},
		{"flac", pad([]byte("fLaC")), AudioFormatFLAC},
		{"ogg", pad([]byte("OggS")), AudioFormatOGG},
		{"m4a", m4a, AudioFormatM4A},
		{"webm", pad([]byte{0x1A, 0x45, 0xDF, 0xA3}), AudioFormatWEBM},
		{"aac adts", pad([]byte{0xFF, 0xF1}), AudioFormatAAC},
		{"unknown", pad([]byte("nope")), ""},
		{"too short", []byte("ID3"), ""},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := DetectAudioFormat(tc.data); got != tc.expected {
				t.Errorf("expected format %q, got %q", tc.expected, got)
			}
		})
	}
}

func TestBaseTranscriptionResponse(t *testing.T) {
	usage := NewTranscriptionUsage(10, 20, 30, 4.2)
	res := NewTranscriptionResponse("hello", "en", usage)

	if res.Text() != "hello" {
		t.Errorf("expected text 'hello', got %q", res.Text())
	}
	if res.Language() != "en" {
		t.Errorf("expected language 'en', got %q", res.Language())
	}
	if res.Usage().InputTokens() != 10 {
		t.Errorf("expected 10 input tokens, got %d", res.Usage().InputTokens())
	}
	if res.Usage().OutputTokens() != 20 {
		t.Errorf("expected 20 output tokens, got %d", res.Usage().OutputTokens())
	}
	if res.Usage().TotalTokens() != 30 {
		t.Errorf("expected 30 total tokens, got %d", res.Usage().TotalTokens())
	}
	if res.Usage().AudioSeconds() != 4.2 {
		t.Errorf("expected 4.2 audio seconds, got %f", res.Usage().AudioSeconds())
	}
}

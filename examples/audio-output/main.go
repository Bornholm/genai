package main

import (
	"context"
	"encoding/base64"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	_ "github.com/bornholm/genai/llm/provider/all"
	"github.com/bornholm/genai/llm/provider/env"
)

var (
	envFile string = ".env"
)

func init() {
	flag.StringVar(&envFile, "env-file", envFile, "client configuration environment file")
}

func main() {
	flag.Parse()
	ctx := context.Background()

	client, err := provider.Create(ctx, env.With("GENAI_", envFile))
	if err != nil {
		log.Fatalf("[FATAL] %+v", err)
	}

	log.Println("[INFO] Requesting audio generation from google/lyria-3-pro-preview...")
	log.Println("[INFO] Make sure GENAI_MODEL=google/lyria-3-pro-preview is set in your .env file")

	stream, err := client.ChatCompletionStream(ctx,
		llm.WithMessages(
			llm.NewMessage(llm.RoleUser,
				`
Song about the joy of stray walking in a deep misty forest. Make it 3 minutes long.

Core: Thunderous, down-tuned eight-string guitar riffs locked in with double-kick blast beats, layered beneath harsh Nordic screamed vocals trading call-and-response with deep, resonant clean male choir harmonies sung in Old Norse cadence. A bowed tagelharpa drones beneath everything, anchoring the tonality in open fifths.
Mashup: The tagelharpa drone and choir harmonics are fused into crushing, distortion-soaked wall-of-sound power chords that swell and recede like tidal waves, while the clean vocal harmonies merge with reverb-drenched delay trails that stretch into cathedral-scale ambient layers, blurring the line between acoustic ritual and full-band siege.
Atmosphere: Howling boreal wind sweeping across a frozen fjord, distant war horns echoing off mountain stone, the crack and roar of a massive longship hull breaking through sea ice, crackling ceremonial bonfires, and the rhythmic stomp of boots and shield-on-shield pounding that locks into the snare pattern like a gathering warband march.
Vibe: Ritualistic, ferocious, and mythic — the feeling of standing on a cliff edge at the world's northern boundary during a blood-red midnight sun, simultaneously mourning the dead and summoning something ancient and untameable from beneath the roots of the world tree. Equal parts grief, glory, and primal defiance.

180 BPM half-time feel · key of D minor · 6/8 to 4/4 time shift at chorus · cinematic build to cathartic peak
`),
		),
		llm.WithModalities("text", "audio"),
		llm.WithAudioOutput("", "wav"),
	)
	if err != nil {
		log.Fatalf("[FATAL] %+v", err)
	}

	var audioChunks []string
	var transcriptChunks []string

	fmt.Print("[AI] ")

	for chunk := range stream {
		if err := chunk.Error(); err != nil {
			log.Printf("[ERROR] %v", err)
			continue
		}

		if delta := chunk.Delta(); delta != nil {
			if content := delta.Content(); content != "" {
				fmt.Print(content)
			}

			if audioDelta, ok := delta.(interface{ AudioData() string }); ok {
				if data := audioDelta.AudioData(); data != "" {
					audioChunks = append(audioChunks, data)
				}
			}
			if audioDelta, ok := delta.(interface{ Transcript() string }); ok {
				if transcript := audioDelta.Transcript(); transcript != "" {
					transcriptChunks = append(transcriptChunks, transcript)
				}
			}
		}

		if chunk.IsComplete() {
			fmt.Println()

			if len(audioChunks) > 0 {
				fullAudio, err := base64.StdEncoding.DecodeString(
					strings.Join(audioChunks, ""))
				if err != nil {
					log.Printf("[ERROR] Failed to decode audio: %v", err)
				} else {
					err = os.WriteFile("output.wav", fullAudio, 0644)
					if err != nil {
						log.Printf("[ERROR] Failed to write audio file: %v", err)
					} else {
						log.Printf("[INFO] Audio saved to output.wav (%d bytes)", len(fullAudio))
					}
				}
			}

			if len(transcriptChunks) > 0 {
				fmt.Printf("[TRANSCRIPT] %s\n", strings.Join(transcriptChunks, ""))
			}

			if usage := chunk.Usage(); usage != nil {
				log.Printf("[USAGE] %d total tokens", usage.TotalTokens())
			}
		}
	}
}

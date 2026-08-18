package provider_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider"
	"github.com/bornholm/genai/llm/provider/minimax"
	"github.com/bornholm/genai/llm/provider/openai"
	"github.com/bornholm/genai/llm/provider/openrouter"
)

// pngPixel est un PNG 1x1 valide : les providers détectent le type de média
// sur les octets décodés, il faut donc de vrais octets d'image.
var pngPixel = []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x0d\n\x2d\xb4\x00\x00\x00\x00IEND\xaeB`\x82")

func pngBase64() string { return base64.StdEncoding.EncodeToString(pngPixel) }

// Chaque provider parle son dialecte sur le fil, mais tous rendent la même
// chose à l'appelant : des octets et un type de média. C'est le contrat de
// l'interface — le test le vérifie pour les trois, requête comprise.
func TestImageGenerationProviders(t *testing.T) {
	cases := []struct {
		name         string
		response     func() any
		option       func(serverURL string) provider.OptionFunc
		verifyBody   func(t *testing.T, body map[string]any)
		expectedType string
	}{
		{
			name: "openai",
			response: func() any {
				return map[string]any{
					"created": 1,
					"data":    []map[string]any{{"b64_json": pngBase64(), "revised_prompt": "a red apple, studio light"}},
				}
			},
			option: func(serverURL string) provider.OptionFunc {
				return provider.WithImageGeneration(openai.Name, openai.Options{
					CommonOptions: provider.CommonOptions{Model: "gpt-image-1", APIKey: "sk-test", BaseURL: serverURL},
				})
			},
			verifyBody: func(t *testing.T, body map[string]any) {
				if body["model"] != "gpt-image-1" || body["prompt"] != "a red apple" {
					t.Errorf("requête openai = %v", body)
				}
				if body["response_format"] != "b64_json" {
					t.Errorf("response_format = %v, attendu b64_json (jamais d'URL éphémère)", body["response_format"])
				}
			},
			expectedType: "image/png",
		},
		{
			name: "openrouter",
			response: func() any {
				return map[string]any{
					"created": 1,
					"data":    []map[string]any{{"b64_json": pngBase64(), "media_type": "image/png"}},
					"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 4000, "total_tokens": 4010},
				}
			},
			option: func(serverURL string) provider.OptionFunc {
				return provider.WithImageGeneration(openrouter.Name, openrouter.Options{
					CommonOptions: provider.CommonOptions{Model: "bytedance-seed/seedream-4.5", APIKey: "sk-test", BaseURL: serverURL},
				})
			},
			verifyBody: func(t *testing.T, body map[string]any) {
				if body["model"] != "bytedance-seed/seedream-4.5" || body["prompt"] != "a red apple" {
					t.Errorf("requête openrouter = %v", body)
				}
			},
			expectedType: "image/png",
		},
		{
			name: "minimax",
			response: func() any {
				return map[string]any{
					"data":      map[string]any{"image_base64": []string{pngBase64()}},
					"base_resp": map[string]any{"status_code": 0, "status_msg": "success"},
				}
			},
			option: func(serverURL string) provider.OptionFunc {
				return provider.WithImageGeneration(minimax.Name, minimax.Options{
					CommonOptions: provider.CommonOptions{Model: "image-01", APIKey: "sk-test", BaseURL: serverURL},
				})
			},
			verifyBody: func(t *testing.T, body map[string]any) {
				if body["model"] != "image-01" || body["prompt"] != "a red apple" {
					t.Errorf("requête minimax = %v", body)
				}
				if body["response_format"] != "base64" {
					t.Errorf("response_format = %v, attendu base64", body["response_format"])
				}
			},
			expectedType: "image/png",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var requestBody map[string]any

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(raw, &requestBody)
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(tc.response())
			}))
			t.Cleanup(server.Close)

			client, err := provider.Create(context.Background(), tc.option(server.URL))
			if err != nil {
				t.Fatalf("provider.Create: %v", err)
			}

			generator, ok := client.(llm.ImageGenerationClient)
			if !ok {
				t.Fatalf("le client %T n'expose pas ImageGeneration", client)
			}

			resp, err := generator.ImageGeneration(context.Background(), "a red apple")
			if err != nil {
				t.Fatalf("ImageGeneration: %v", err)
			}

			tc.verifyBody(t, requestBody)

			images := resp.Images()
			if len(images) != 1 {
				t.Fatalf("images = %d, attendu 1", len(images))
			}
			if string(images[0].Data()) != string(pngPixel) {
				t.Error("les octets décodés ne correspondent pas à l'image d'origine")
			}
			if images[0].MediaType() != tc.expectedType {
				t.Errorf("media type = %q, attendu %q", images[0].MediaType(), tc.expectedType)
			}
		})
	}
}

// Une erreur applicative MiniMax arrive avec un HTTP 200 : c'est base_resp
// qui fait foi, jamais le statut HTTP seul.
func TestMinimaxErrorWithHTTP200(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"base_resp":{"status_code":2049,"status_msg":"invalid api key"}}`))
	}))
	t.Cleanup(server.Close)

	client := minimax.NewImageGenerationClient(nil, server.URL, "bad-key", "image-01")

	_, err := client.ImageGeneration(context.Background(), "a red apple")
	if err == nil {
		t.Fatal("une erreur base_resp doit remonter malgré le HTTP 200")
	}
}

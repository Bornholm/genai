package conformance

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

// redSquarePNG generates a 50×50 solid red PNG and returns its base64 encoding.
func redSquarePNG() string {
	img := image.NewRGBA(image.Rect(0, 0, 50, 50))
	red := color.RGBA{R: 255, G: 0, B: 0, A: 255}
	for y := range 50 {
		for x := range 50 {
			img.Set(x, y, red)
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func testMultimodal(t *testing.T, client llm.Client) {
	t.Helper()

	chatClient, ok := client.(llm.ChatCompletionClient)
	if !ok {
		t.Skip("client does not implement ChatCompletionClient")
	}

	ctx := context.Background()

	t.Run("ImageAttachment", func(t *testing.T) {
		att, err := llm.NewImageAttachment("image/png", redSquarePNG(), false)
		if err != nil {
			t.Fatalf("failed to create image attachment: %v", err)
		}

		res, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMultimodalMessage(
					llm.RoleUser,
					"What color is this image? Reply with only the color name.",
					att,
				),
			),
			llm.WithTemperature(0),
		)
		if err != nil {
			t.Fatalf("ChatCompletion error: %v", err)
		}

		content := strings.ToLower(res.Message().Content())
		if !strings.Contains(content, "red") {
			t.Errorf("expected response to mention 'red', got: %q", res.Message().Content())
		}
	})
}

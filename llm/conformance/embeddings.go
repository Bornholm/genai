package conformance

import (
	"context"
	"math"
	"testing"

	"github.com/bornholm/genai/llm"
)

func testEmbeddings(t *testing.T, client llm.Client) {
	t.Helper()

	embClient, ok := client.(llm.EmbeddingsClient)
	if !ok {
		t.Skip("client does not implement EmbeddingsClient")
	}

	ctx := context.Background()

	t.Run("VectorShape", func(t *testing.T) {
		res, err := embClient.Embeddings(ctx, []string{"hello world"})
		if err != nil {
			t.Fatalf("Embeddings error: %v", err)
		}
		if len(res.Embeddings()) == 0 {
			t.Fatal("expected at least one embedding vector")
		}
		if len(res.Embeddings()[0]) == 0 {
			t.Error("expected non-empty embedding vector")
		}
		if res.Usage() == nil {
			t.Error("expected non-nil usage")
		}
	})

	t.Run("SimilarityOrdering", func(t *testing.T) {
		// "cat" and "kitten" should be closer to each other than "cat" and "automobile".
		res, err := embClient.Embeddings(ctx, []string{"cat", "kitten", "automobile"})
		if err != nil {
			t.Fatalf("Embeddings error: %v", err)
		}
		if len(res.Embeddings()) < 3 {
			t.Fatalf("expected 3 embeddings, got %d", len(res.Embeddings()))
		}

		simCatKitten := cosineSimilarity(res.Embeddings()[0], res.Embeddings()[1])
		simCatAuto := cosineSimilarity(res.Embeddings()[0], res.Embeddings()[2])

		if simCatKitten <= simCatAuto {
			t.Errorf("expected sim(cat,kitten)=%.4f > sim(cat,automobile)=%.4f", simCatKitten, simCatAuto)
		}
	})
}

func cosineSimilarity(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

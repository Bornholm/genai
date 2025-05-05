package testsuite

import (
	"context"
	"embed"
	"io"
	"testing"
	"unicode"

	"github.com/agnivade/levenshtein"
	"github.com/andreyvit/diff"
	"github.com/bornholm/genai/extract"
	"github.com/davecgh/go-spew/spew"
	"github.com/pkg/errors"

	_ "embed"
)

//go:embed testdata/extract/*
var extractTestData embed.FS

func ExtractText(t *testing.T, factory func() (extract.TextClient, error)) {
	type testCase struct {
		Name string
		Run  func(t *testing.T, client extract.TextClient)
	}

	testCases := []testCase{
		{
			Name: "Simple PDF text extraction",
			Run: func(t *testing.T, client extract.TextClient) {
				ctx := context.TODO()

				pdfFile, err := extractTestData.Open("testdata/extract/test.pdf")
				if err != nil {
					t.Fatalf("%+v", errors.WithStack(err))
				}

				defer pdfFile.Close()

				res, err := client.Text(ctx, extract.WithReader(pdfFile))
				if err != nil {
					t.Fatalf("%+v", errors.WithStack(err))
				}

				t.Logf("Response: %s", spew.Sdump(res))

				output, err := io.ReadAll(res.Output())
				if err != nil {
					t.Fatalf("%+v", errors.WithStack(err))
				}

				markdownFile, err := extractTestData.ReadFile("testdata/extract/test.md")
				if err != nil {
					t.Fatalf("%+v", errors.WithStack(err))
				}

				expected := string(markdownFile)
				got := string(output)

				averageDistance := getAverageLevenshteinDistance(expected, got)

				t.Logf("Average levenshtein distance: %v", averageDistance)

				if averageDistance > 0 {
					t.Errorf("output: \n\n%s", diff.LineDiff(expected, got))
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			client, err := factory()
			if err != nil {
				t.Fatalf("%+v", errors.WithStack(err))
			}

			tc.Run(t, client)
		})
	}
}

func getAverageLevenshteinDistance(t1, t2 string) float64 {
	words1 := splitByWords(t1)
	words2 := splitByWords(t2)

	maxWords := max(len(words1), len(words2))

	totalDistance := 0
	for i := 0; i < maxWords; i++ {
		var w1 string
		if i < len(words1) {
			w1 = words1[i]
		}

		var w2 string
		if i < len(words2) {
			w2 = words2[i]
		}

		totalDistance += levenshtein.ComputeDistance(w1, w2)
	}

	return float64(totalDistance) / float64(maxWords)
}

func splitByWords(text string) []string {
	words := make([]string, 0)

	var word []byte
	for idx, rune := range text {
		if unicode.IsSpace(rune) || unicode.IsPunct(rune) {
			if word != nil {
				words = append(words, string(word))
				word = nil
			}

			continue
		}

		word = append(word, text[idx])
	}

	if word != nil {
		word = append(word, text[len(text)-1])
		words = append(words, string(word))
	}

	return words
}

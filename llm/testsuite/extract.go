package testsuite

import (
	"context"
	"embed"
	"io"
	"strings"
	"testing"

	"github.com/andreyvit/diff"
	"github.com/bornholm/genai/llm"
	"github.com/davecgh/go-spew/spew"
	"github.com/pkg/errors"

	_ "embed"
)

//go:embed testdata/extract/*
var extractTestData embed.FS

func ExtractText(t *testing.T, factory func() (llm.ExtractTextClient, error)) {
	type testCase struct {
		Name string
		Run  func(t *testing.T, client llm.ExtractTextClient)
	}

	testCases := []testCase{
		{
			Name: "Simple PDF text extraction",
			Run: func(t *testing.T, client llm.ExtractTextClient) {
				ctx := context.TODO()

				pdfFile, err := extractTestData.Open("testdata/extract/test.pdf")
				if err != nil {
					t.Fatalf("%+v", errors.WithStack(err))
				}

				defer pdfFile.Close()

				res, err := client.ExtractText(ctx, llm.WithReader(pdfFile))
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

				expected := strings.TrimSpace(string(markdownFile))
				got := strings.TrimSpace(string(output))

				if e, g := expected, got; e != g {
					t.Errorf("output: %s", diff.LineDiff(e, g))
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

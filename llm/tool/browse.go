package tool

import (
	"context"
	"strings"

	md "github.com/JohannesKaufmann/html-to-markdown"
	"github.com/bornholm/genai/llm"
	"github.com/gocolly/colly"
	"github.com/pkg/errors"
)

func Browse() *llm.FuncTool {
	return llm.NewFuncTool(
		"browse",
		"Open the given webpage url and return its content as markdown",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"url": map[string]string{
					"type":        "string",
					"description": "The URL to open",
				},
			},
			"required": []string{"url"},
		},
		func(ctx context.Context, params map[string]any) (string, error) {
			url, err := Param[string](params, "url")
			if err != nil {
				return "", errors.WithStack(err)
			}

			var sb strings.Builder

			collector := colly.NewCollector()

			converter := md.NewConverter("", true, nil)

			collector.OnHTML("body", func(h *colly.HTMLElement) {
				markdown := converter.Convert(h.DOM)
				sb.WriteString(markdown)
			})

			collector.Visit(url)

			return sb.String(), nil
		},
	)
}

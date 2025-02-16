package tool

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/gocolly/colly"
	"github.com/pkg/errors"
)

func WebSearch() *llm.FuncTool {
	return llm.NewFuncTool(
		"web_search",
		"Search webpages about the given topic",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"topic": map[string]string{
					"type":        "string",
					"description": "The searched topic",
				},
			},
			"required": []string{"topic"},
		},
		func(ctx context.Context, params map[string]any) (string, error) {
			topic, err := Param[string](params, "topic")
			if err != nil {
				return "", errors.WithStack(err)
			}

			url := &url.URL{
				Scheme: "https",
				Host:   "duckduckgo.com",
				Path:   "/html/",
			}

			query := url.Query()
			query.Add("q", fmt.Sprintf("%s", topic))

			url.RawQuery = query.Encode()

			log.Printf("Search with '%s'", url.String())

			var sb strings.Builder

			collector := colly.NewCollector()

			sb.WriteString("# Results\n\n")

			collector.OnHTML(".result", func(h *colly.HTMLElement) {
				title := strings.TrimSpace(h.DOM.Find(".result__title").Text())
				if title == "" {
					return
				}

				rawLink := h.DOM.Find(".result__a").AttrOr("href", "")
				if rawLink == "" {
					return
				}

				link, err := url.Parse(rawLink)
				if err != nil {
					return
				}

				snippet := strings.TrimSpace(h.DOM.Find(".result__snippet").Text())
				if snippet == "" {
					return
				}

				sb.WriteString("## ")
				sb.WriteString(title)
				sb.WriteString("\n\n")
				sb.WriteString(snippet)
				sb.WriteString("\n\nLink: ")
				sb.WriteString(link.Query().Get("uddg"))
				sb.WriteString("\n\n")
			})

			collector.Visit(url.String())

			return sb.String(), nil
		},
	)
}

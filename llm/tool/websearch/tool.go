package websearch

import (
	"context"
	"fmt"
	"log/slog"
	"net/url"
	"strings"

	md "github.com/JohannesKaufmann/html-to-markdown"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/tool"
	"github.com/davecgh/go-spew/spew"
	"github.com/gocolly/colly"
	"github.com/pkg/errors"
)

const defaultSystemPrompt = `
You are an intelligent agent specialized in synthesizing information from a large corpus of documents. Your primary task is to generate concise and coherent summaries based on a given query. Here's how you should approach this task:

1. **Understand the Query:** Carefully read and comprehend the query provided. Identify the key themes, questions, or topics that need to be addressed in your synthesis.

2. **Analyze the Documents:** Scan through the corpus of documents to identify relevant sections that pertain to the query. Look for key points, arguments, data, and conclusions that are directly related to the query.

3. **Extract Key Information:** From the relevant sections, extract the most important information. This includes main ideas, supporting details, statistics, and any other pertinent data.

4. **Synthesize Information:** Combine the extracted information into a coherent summary. Ensure that the summary addresses the query comprehensively and logically flows from one point to the next.

5. **Maintain Clarity and Conciseness:** Write the synthesis in clear and concise language. Avoid jargon and ensure that the summary is easy to understand for the intended audience.

6. **Cite Sources**: Where appropriate, cite the original documents to maintain the integrity and credibility of the synthesis.

7. **Review and Refine**: After drafting the synthesis, review it for accuracy, coherence, and completeness. Make any necessary revisions to improve the quality of the summary.

Example Query:
"Summarize the key findings on the impact of climate change on global agriculture from the provided research papers."

Example Synthesis:
"The research papers highlight several key findings on the impact of climate change on global agriculture. Increasing temperatures and changing precipitation patterns are leading to reduced crop yields in many regions. Extreme weather events, such as droughts and floods, are becoming more frequent, further threatening food security. However, some studies suggest that innovative farming techniques and climate-resilient crop varieties could mitigate these impacts. Overall, the papers underscore the urgent need for adaptation strategies to sustain global agriculture in the face of climate change."
`

const defaultUserPromptTemplate = `
Write the synthesis of the following articles based on this query. Do not output anything else.

## Query

{{ .Query }}

## Articles

{{ range $url, $content := .Articles }}
### {{ $url }}

{{ $content }}

{{ end }}
`

type Options struct {
	ChatCompletionOptions []llm.ChatCompletionOptionFunc
	SystemPrompt          string
	UserPromptTemplate    string
	CollectLinks          func(ctx context.Context, topic string) ([]string, error)
}

type OptionFunc func(opts *Options)

func WithDuckDuckGoSearch(maxLinks int) OptionFunc {
	return func(opts *Options) {
		opts.CollectLinks = func(ctx context.Context, topic string) ([]string, error) {
			url := &url.URL{
				Scheme: "https",
				Host:   "duckduckgo.com",
				Path:   "/html/",
			}

			query := url.Query()
			query.Add("q", fmt.Sprintf("%s", topic))

			url.RawQuery = query.Encode()

			collector := colly.NewCollector()

			links := make([]string, 0)

			collector.OnHTML(".result", func(h *colly.HTMLElement) {
				rawLink := h.DOM.Find(".result__a").AttrOr("href", "")
				if rawLink == "" {
					return
				}

				link, err := url.Parse(rawLink)
				if err != nil {
					return
				}

				links = append(links, link.Query().Get("uddg"))
			})

			collector.Visit(url.String())

			if len(links) > maxLinks {
				links = links[:maxLinks-1]
			}

			return links, nil
		}
	}
}

func WithChatCompletionsOptions(funcs ...llm.ChatCompletionOptionFunc) OptionFunc {
	return func(opts *Options) {
		opts.ChatCompletionOptions = funcs
	}
}

func NewOptions(funcs ...OptionFunc) *Options {
	opts := &Options{
		ChatCompletionOptions: make([]llm.ChatCompletionOptionFunc, 0),
		UserPromptTemplate:    defaultUserPromptTemplate,
		SystemPrompt:          defaultSystemPrompt,
	}
	WithDuckDuckGoSearch(5)(opts)
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

func Tool(client llm.Client, funcs ...OptionFunc) *llm.FuncTool {
	opts := NewOptions(funcs...)
	return llm.NewFuncTool(
		"websearch",
		"Execute a research about a given topic on the web and retrieve a synthesis of the informations found",
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
			topic, err := tool.Param[string](params, "topic")
			if err != nil {
				return "", errors.WithStack(err)
			}

			links, err := opts.CollectLinks(ctx, topic)
			if err != nil {
				return "", errors.Wrap(err, "could not retrieve links about the topic")
			}

			articles := make(map[string]string)
			for _, l := range links {
				content, err := loadURL(l)
				if err != nil {
					slog.ErrorContext(ctx, "could not load link", slog.Any("errors", errors.WithStack(err)))
					continue
				}

				articles[l] = content
			}

			options := []llm.ChatCompletionOptionFunc{}
			options = append(options, opts.ChatCompletionOptions...)

			userPrompt, err := llm.PromptTemplate(opts.UserPromptTemplate, struct {
				Query    string
				Articles map[string]string
			}{
				Query:    topic,
				Articles: articles,
			})
			if err != nil {
				return "", errors.Wrap(err, "could not compile user prompte template")
			}

			options = append(options, llm.WithMessages(
				llm.NewMessage(llm.RoleSystem, opts.SystemPrompt),
				llm.NewMessage(llm.RoleUser, userPrompt),
			))

			spew.Dump(userPrompt)

			res, err := client.ChatCompletion(ctx, options...)
			if err != nil {
				return "", errors.Wrap(err, "could not generate synthesis of topic informations")
			}

			return fmt.Sprintf("**Search Result:**\n\n%s", res.Message().Content()), nil
		},
	)
}

func loadURL(url string) (string, error) {
	collector := colly.NewCollector()

	converter := md.NewConverter("", true, nil)

	var sb strings.Builder

	collector.OnHTML("body", func(h *colly.HTMLElement) {
		markdown := converter.Convert(h.DOM)
		sb.WriteString(markdown)
	})

	if err := collector.Visit(url); err != nil {
		return "", errors.WithStack(err)
	}

	return sb.String(), nil
}

package index

import (
	"context"
	"log"
	"strconv"
	"strings"

	"github.com/blevesearch/bleve/v2"
	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/tool"
	blevellm "github.com/bornholm/genai/llm/tool/index/bleve"
	"github.com/pkg/errors"
)

type SearchOptions struct {
	Resources           []Resource
	ResourceCollections []ResourceCollection
	ResultTransformer   ResultTransformer
}

type ResultTransformer interface {
	Transform(ctx context.Context, result *bleve.SearchResult) (string, error)
}

type TransformFunc func(ctx context.Context, result *bleve.SearchResult) (string, error)

func (f TransformFunc) Transform(ctx context.Context, result *bleve.SearchResult) (string, error) {
	return f(ctx, result)
}

func WithResultTransformer(t ResultTransformer) SearchOptionFunc {
	return func(opts *SearchOptions) {
		opts.ResultTransformer = t
	}
}

type SearchOptionFunc func(opts *SearchOptions)

func WithResources(resources ...Resource) SearchOptionFunc {
	return func(opts *SearchOptions) {
		opts.Resources = resources
	}
}

func WithResourceCollections(collections ...ResourceCollection) SearchOptionFunc {
	return func(opts *SearchOptions) {
		opts.ResourceCollections = collections
	}
}

func NewSearchOptions(funcs ...SearchOptionFunc) *SearchOptions {
	opts := &SearchOptions{
		Resources:           make([]Resource, 0),
		ResourceCollections: make([]ResourceCollection, 0),
		ResultTransformer:   DefaultResultTransform,
	}

	for _, fn := range funcs {
		fn(opts)
	}

	return opts
}

func Search(funcs ...SearchOptionFunc) (*llm.FuncTool, error) {
	opts := NewSearchOptions(funcs...)

	mapping := bleve.NewIndexMapping()

	mapping.TypeField = "_type"
	mapping.DefaultAnalyzer = blevellm.AnalyzerDynamicLang

	resourceMapping := bleve.NewDocumentMapping()

	contentFieldMapping := bleve.NewTextFieldMapping()
	contentFieldMapping.Store = true
	contentFieldMapping.Analyzer = blevellm.AnalyzerDynamicLang
	resourceMapping.AddFieldMappingsAt("content", contentFieldMapping)

	sourceFieldMapping := bleve.NewTextFieldMapping()
	sourceFieldMapping.Analyzer = blevellm.AnalyzerDynamicLang
	sourceFieldMapping.Store = true
	resourceMapping.AddFieldMappingsAt("source", sourceFieldMapping)

	mapping.AddDocumentMapping("resource", resourceMapping)

	index, err := bleve.NewMemOnly(mapping)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	resources := []Resource{}
	resources = append(resources, opts.Resources...)
	for _, c := range opts.ResourceCollections {
		res, err := c.Resources()
		if err != nil {
			return nil, errors.Wrap(err, "could not collect resources")
		}

		resources = append(resources, res...)
	}

	for _, r := range resources {
		content, err := r.Content()
		if err != nil {
			return nil, errors.Wrapf(err, "could not retrieve resource '%s' content", r.ID())
		}

		log.Printf("Indexing resource '%s'", r.ID())

		if err := index.Index(r.ID(), map[string]string{
			"_type":   "resource",
			"content": content,
			"source":  r.Source(),
		}); err != nil {
			return nil, errors.Wrapf(err, "could not retrieve resource '%s' content", r.ID())
		}
	}

	tool := llm.NewFuncTool(
		"index_search",
		"Search informations about the given topic in the index",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"topic": map[string]string{
					"type":        "string",
					"description": "The searched topic",
				},
			},
			"required":             []string{"topic"},
			"additionalProperties": false,
		},
		func(ctx context.Context, params map[string]any) (string, error) {
			topic, err := tool.Param[string](params, "topic")
			if err != nil {
				return "", errors.Wrap(err, "invalid parameter 'topic'")
			}

			req := bleve.NewSearchRequest(bleve.NewQueryStringQuery(topic))
			req.Fields = []string{"content", "source"}
			req.Size = 2

			result, err := index.SearchInContext(ctx, req)
			if err != nil {
				return "", errors.Wrap(err, "could not execute search")
			}

			transformed, err := opts.ResultTransformer.Transform(ctx, result)
			if err != nil {
				return "", errors.Wrap(err, "could not transform search result")
			}

			return transformed, nil
		},
	)

	return tool, nil
}

var DefaultResultTransform = TransformFunc(func(ctx context.Context, result *bleve.SearchResult) (string, error) {
	var sb strings.Builder

	sb.WriteString("## Search Results\n\n")

	for idx, r := range result.Hits {
		rawSource, exists := r.Fields["source"]
		if !exists {
			continue
		}

		rawContent, exists := r.Fields["content"]
		if !exists {
			continue
		}

		sb.WriteString("### ")
		sb.WriteString(strconv.FormatInt(int64(idx+1), 10))
		sb.WriteString(". ")
		sb.WriteString(r.ID)
		sb.WriteString(" (source: ")
		sb.WriteString(rawSource.(string))
		sb.WriteString(")\n\n")
		sb.WriteString(rawContent.(string))
		sb.WriteString("\n\n")
	}

	return sb.String(), nil
})

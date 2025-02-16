package index

import (
	"net/url"
	"os"
	"path/filepath"
	"strings"

	md "github.com/JohannesKaufmann/html-to-markdown"
	"github.com/PuerkitoBio/goquery"
	"github.com/gocolly/colly"
	"github.com/pkg/errors"
	"github.com/yargevad/filepathx"
)

type Resource interface {
	ID() string
	Content() (string, error)
	Source() string
}

type ResourceCollection interface {
	Resources() ([]Resource, error)
}

type globResourceCollection struct {
	pattern string
}

// Resources implements ResourceCollection.
func (c *globResourceCollection) Resources() ([]Resource, error) {
	files, err := filepathx.Glob(c.pattern)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	resources := []Resource{}
	for _, f := range files {
		resources = append(resources, URLResource(f))
	}

	return resources, nil
}

var _ ResourceCollection = &globResourceCollection{}

func FileCollection(pattern string) ResourceCollection {
	return &globResourceCollection{pattern: pattern}
}

type scraperCollection struct {
	domain string
}

// Resources implements ResourceCollection.
func (c *scraperCollection) Resources() ([]Resource, error) {
	url, err := url.Parse(c.domain)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	resources := []Resource{}

	collector := colly.NewCollector()

	collector.AllowURLRevisit = false
	collector.AllowedDomains = []string{url.Hostname()}
	collector.Async = false

	collector.OnHTML("body", func(h *colly.HTMLElement) {
		resources = append(resources, URLResource(h.Request.URL.String()))

		h.DOM.Find("a[href]").Each(func(i int, s *goquery.Selection) {
			link, exists := s.Attr("href")
			if !exists || link == "" {
				return
			}

			h.Request.Visit(h.Request.AbsoluteURL(link))
		})
	})

	if err := collector.Visit(c.domain); err != nil {
		return nil, errors.Wrap(err, "could not scrape website")
	}

	return resources, nil
}

var _ ResourceCollection = &scraperCollection{}

func WebsiteCollection(domain string) ResourceCollection {
	return &scraperCollection{domain: domain}
}

var urlLoaders = map[string]ResourceLoader{}

func AddResourceLoader(scheme string, loader ResourceLoader) {
	urlLoaders[scheme] = loader
}

type URLResource string

// Content implements Resource.
func (r URLResource) Content() (string, error) {
	url, err := url.Parse(string(r))
	if err != nil {
		return "", errors.WithStack(err)
	}

	loader, exists := urlLoaders[url.Scheme]
	if !exists {
		return "", errors.Errorf("url loader '%s' not found", url.Scheme)
	}

	content, err := loader.Load(url)
	if err != nil {
		return "", errors.WithStack(err)
	}

	return content, nil
}

// ID implements Resource.
func (r URLResource) ID() string {
	return string(r)
}

// Source implements Resource.
func (r URLResource) Source() string {
	return string(r)
}

var _ Resource = URLResource("")

type ResourceLoader interface {
	Load(u *url.URL) (string, error)
}

type ResourceLoaderFunc func(*url.URL) (string, error)

func (f ResourceLoaderFunc) Load(u *url.URL) (string, error) {
	return f(u)
}

var HTTPResourceLoader ResourceLoaderFunc = func(u *url.URL) (string, error) {
	var sb strings.Builder

	collector := colly.NewCollector()

	converter := md.NewConverter("", true, nil)

	collector.OnHTML("body", func(h *colly.HTMLElement) {
		markdown := converter.Convert(h.DOM)
		sb.WriteString(markdown)
	})

	collector.Visit(u.String())

	return sb.String(), nil
}

var FileResourceLoader ResourceLoaderFunc = func(u *url.URL) (string, error) {
	filename := filepath.Join(u.Hostname(), u.Path)

	data, err := os.ReadFile(filename)
	if err != nil {
		return "", errors.Wrapf(err, "could not read file '%s'", data)
	}

	return string(data), nil
}

func init() {
	AddResourceLoader("http", HTTPResourceLoader)
	AddResourceLoader("https", HTTPResourceLoader)
	AddResourceLoader("file", FileResourceLoader)
	AddResourceLoader("", FileResourceLoader)
}

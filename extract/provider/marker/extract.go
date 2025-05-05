package marker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/bornholm/genai/extract"
	"github.com/pkg/errors"
)

type TextClient struct {
	baseURL *url.URL
	http    *http.Client
}

// Text implements llm.TextClient.
func (c *TextClient) Text(ctx context.Context, funcs ...extract.TextOptionFunc) (extract.TextResponse, error) {
	opts, err := extract.NewTextOptions(funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if opts.Reader == nil {
		return nil, errors.WithStack(extract.ErrMissingReader)
	}

	if closer, ok := opts.Reader.(io.Closer); ok {
		defer closer.Close()
	}

	endpoint := c.baseURL.JoinPath("/marker/upload")

	var body bytes.Buffer
	form := multipart.NewWriter(&body)

	if err := form.WriteField("output_format", "markdown"); err != nil {
		return nil, errors.WithStack(err)
	}

	var filename string
	if opts.Filename != "" {
		filename = opts.Filename
	} else {
		filename = fmt.Sprintf("file_%d", time.Now().Nanosecond())
	}

	fileWriter, err := form.CreateFormFile("file", filename)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if _, err := io.Copy(fileWriter, opts.Reader); err != nil {
		return nil, errors.WithStack(err)
	}

	if err := form.Close(); err != nil {
		return nil, errors.WithStack(err)
	}

	if len(opts.Pages) > 0 {
		var sb strings.Builder
		for i, p := range opts.Pages {
			if i > 0 {
				sb.WriteString(",")
			}
			sb.WriteString(strconv.FormatInt(int64(p), 10))
		}

		if err := form.WriteField("page_range", sb.String()); err != nil {
			return nil, errors.WithStack(err)
		}
	}

	req, err := http.NewRequest("POST", endpoint.String(), &body)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	req = req.WithContext(ctx)

	req.Header.Set("Content-Type", form.FormDataContentType())

	httpRes, err := c.http.Do(req)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	defer httpRes.Body.Close()

	httpBody, err := io.ReadAll(httpRes.Body)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if httpRes.StatusCode != http.StatusOK {
		return nil, errors.Errorf("unexpected http response '%d' (%s):\n%s", httpRes.StatusCode, httpRes.Status, httpBody)
	}

	type markerResponse struct {
		Format   string            `json:"format"`
		Output   string            `json:"output"`
		Images   map[string]string `json:"images"`
		Metadata any               `json:"metadata"`
		Success  bool              `json:"success"`
	}

	var markerRes markerResponse

	if err := json.Unmarshal(httpBody, &markerRes); err != nil {
		return nil, errors.WithStack(err)
	}

	if markerRes.Success == false {
		return nil, errors.Errorf("transformation failed:\n%s", httpBody)
	}

	return &TextResponse{
		format:   extract.TextFormatMarkdown,
		output:   bytes.NewBufferString(markerRes.Output),
		Images:   markerRes.Images,
		Metadata: markerRes.Metadata,
	}, nil
}

func NewTextClient(baseURL *url.URL) *TextClient {
	return &TextClient{
		baseURL: baseURL,
		http: &http.Client{
			Timeout: 0,
		},
	}
}

var _ extract.TextClient = &TextClient{}

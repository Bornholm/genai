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

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type ExtractTextClient struct {
	baseURL *url.URL
	http    *http.Client
}

// ExtractText implements llm.ExtractTextClient.
func (c *ExtractTextClient) ExtractText(ctx context.Context, funcs ...llm.ExtractTextOptionFunc) (llm.ExtractTextResponse, error) {
	opts, err := llm.NewExtractTextOptions(funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if opts.Reader == nil {
		return nil, errors.WithStack(llm.ErrMissingReader)
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

	return &ExtractTextResponse{
		format:   llm.ExtractTextFormatMarkdown,
		output:   bytes.NewBufferString(markerRes.Output),
		Images:   markerRes.Images,
		Metadata: markerRes.Metadata,
	}, nil
}

func NewExtractTextClient(baseURL *url.URL) *ExtractTextClient {
	return &ExtractTextClient{
		baseURL: baseURL,
		http: &http.Client{
			Timeout: 0,
		},
	}
}

var _ llm.ExtractTextClient = &ExtractTextClient{}

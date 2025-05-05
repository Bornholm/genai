package mistral

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/bornholm/genai/extract"
	"github.com/pkg/errors"
)

type TextClient struct {
	apiKey  string
	baseURL *url.URL
	http    *http.Client
}

// Text implements llm.TextClient.
func (c *TextClient) Text(ctx context.Context, funcs ...extract.TextOptionFunc) (extract.TextResponse, error) {
	opts, err := extract.NewTextOptions(funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	fileUpload, err := c.uploadFile(ctx, opts)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	defer func() {
		fileDelete, err := c.deleteFile(ctx, fileUpload.ID)
		if err != nil {
			slog.ErrorContext(ctx, "could not delete file", slog.Any("error", errors.WithStack(err)), slog.Any("fileID", fileUpload.ID))
		}

		if !fileDelete.Deleted {
			slog.WarnContext(ctx, "file was not deleted", slog.Any("id", errors.WithStack(err)), slog.Any("fileID", fileUpload.ID))
		}
	}()

	signedURL, err := c.getSignedURL(ctx, fileUpload.ID)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	ocr, err := c.executeOCR(ctx, fileUpload.Filename, signedURL.URL)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	var sb strings.Builder
	for _, p := range ocr.Pages {
		sb.WriteString(p.Markdown)
	}

	return &TextResponse{
		format: extract.TextFormatMarkdown,
		output: bytes.NewBufferString(sb.String()),
	}, nil
}

// JSON response ref
//
//	{
//		"id": "497f6eca-6276-4993-bfeb-53cbbbba6f09",
//		"object": "file",
//		"bytes": 13000,
//		"created_at": 1716963433,
//		"filename": "files_upload.jsonl",
//		"purpose": "fine-tune",
//		"sample_type": "pretrain",
//		"num_lines": 0,
//		"source": "upload"
//		}
type fileUploadResponse struct {
	ID         string `json:"id"`
	Object     string `json:"object"`
	Bytes      int    `json:"bytes"`
	CreatedAt  int    `json:"created_at"`
	Filename   string `json:"filename"`
	SampleType string `json:"sample_type"`
	Purpose    string `json:"purpose"`
	NumLines   int    `json:"num_lines"`
	Source     string `json:"source"`
}

func (c *TextClient) uploadFile(ctx context.Context, opts *extract.TextOptions) (*fileUploadResponse, error) {
	if opts.Reader == nil {
		return nil, errors.WithStack(extract.ErrMissingReader)
	}

	if closer, ok := opts.Reader.(io.Closer); ok {
		defer closer.Close()
	}

	var body bytes.Buffer
	form := multipart.NewWriter(&body)

	if err := form.WriteField("purpose", "ocr"); err != nil {
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

	header := http.Header{}
	header.Set("Content-Type", form.FormDataContentType())

	var res fileUploadResponse
	if err := c.jsonRequest(ctx, "POST", "/v1/files", header, &body, &res); err != nil {
		return nil, errors.WithStack(err)
	}

	return &res, nil
}

// JSON response ref
//
//	{
//	  "id": "497f6eca-6276-4993-bfeb-53cbbbba6f09",
//	  "object": "file",
//	  "deleted": false
//	}
type deleteFileResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

func (c *TextClient) deleteFile(ctx context.Context, id string) (*deleteFileResponse, error) {
	var res deleteFileResponse
	if err := c.jsonRequest(ctx, "DELETE", fmt.Sprintf("/v1/files/%s", id), nil, nil, &res); err != nil {
		return nil, errors.WithStack(err)
	}

	return &res, nil
}

// JSON response ref
//
//	{
//			 "url": "string"
//	}
type signedURLResponse struct {
	URL string `json:"url"`
}

func (c *TextClient) getSignedURL(ctx context.Context, id string) (*signedURLResponse, error) {
	var res signedURLResponse
	if err := c.jsonRequest(ctx, "GET", fmt.Sprintf("/v1/files/%s/url", id), nil, nil, &res); err != nil {
		return nil, errors.WithStack(err)
	}

	return &res, nil
}

// JSON request ref
//
//	{
//	  "model": "string",
//	  "id": "string",
//	  "document": {
//	    "document_url": "string",
//	    "document_name": "string",
//	    "type": "document_url"
//	  },
//	  "pages": [
//	    0
//	  ],
//	  "include_image_base64": true,
//	  "image_limit": 0,
//	  "image_min_size": 0
//	}
type ocrResquest struct {
	Model              string      `json:"model"`
	ID                 string      `json:"id,omitempty"`
	Document           ocrDocument `json:"document"`
	Pages              []int       `json:"pages,omitempty"`
	IncludeImageBase64 bool        `json:"include_image_base64"`
	ImageLimit         int         `json:"image_limit,omitempty"`
	ImageMinSize       int         `json:"image_min_size,omitempty"`
}

type ocrDocument struct {
	DocumentURL  string `json:"document_url"`
	DocumentName string `json:"document_name,omitempty"`
	Type         string `json:"type"`
}

// JSON response ref
//
//	{
//		"pages": [
//		{}
//		],
//		"model": "string",
//		"usage_info": {
//		"pages_processed": 0,
//		"doc_size_bytes": 0
//		}
//	}
type ocrResponse struct {
	Pages     []ocrPage      `json:"pages"`
	Model     string         `json:"model"`
	UsageInfo map[string]any `json:"usage_info"`
}

type ocrPage struct {
	Index      int           `json:"index"`
	Markdown   string        `json:"markdown"`
	Images     []ocrImage    `json:"images"`
	Dimensions ocrDimensions `json:"dimensions"`
}

type ocrImage struct {
	ID           string `json:"id"`
	TopLeftX     int    `json:"top_left_x"`
	TopLeftY     int    `json:"top_left_y"`
	BottomRightX int    `json:"bottom_right_x"`
	BottomLeftX  int    `json:"bottom_left_x"`
	ImageBase64  string `json:"image_base64"`
}

type ocrDimensions struct {
	DPI    int `json:"dpi"`
	Height int `json:"height"`
	Width  int `json:"int"`
}

func (c *TextClient) executeOCR(ctx context.Context, filename string, fileURL string) (*ocrResponse, error) {
	var req ocrResquest

	req.Model = "mistral-ocr-latest"
	req.Document.Type = "document_url"
	req.Document.DocumentURL = fileURL
	req.IncludeImageBase64 = true

	data, err := json.Marshal(req)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	header := http.Header{}
	header.Set("Content-Type", "application/json")

	var res ocrResponse
	if err := c.jsonRequest(ctx, "POST", "/v1/ocr", header, bytes.NewBuffer(data), &res); err != nil {
		return nil, errors.WithStack(err)
	}

	return &res, nil
}

func NewTextClient(baseURL *url.URL, apiKey string) *TextClient {
	return &TextClient{
		apiKey:  apiKey,
		baseURL: baseURL,
		http: &http.Client{
			Timeout: 0,
		},
	}
}

var _ extract.TextClient = &TextClient{}

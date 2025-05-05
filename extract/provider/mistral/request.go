package mistral

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"

	"github.com/pkg/errors"
)

func (c *TextClient) request(ctx context.Context, method string, path string, header http.Header, body io.Reader, result io.Writer) error {
	url, err := url.Parse(path)
	if err != nil {
		return errors.WithStack(err)
	}

	url.Scheme = c.baseURL.Scheme
	url.Host = c.baseURL.Host
	url.User = c.baseURL.User
	url.Path = c.baseURL.JoinPath(path).Path

	slog.DebugContext(ctx, "new request", slog.String("method", method), slog.String("host", url.Host), slog.String("path", path))

	req, err := http.NewRequest(method, url.String(), body)
	if err != nil {
		return errors.WithStack(err)
	}

	req = req.WithContext(ctx)

	req.Header = header
	if req.Header == nil {
		req.Header = http.Header{}
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))

	res, err := c.http.Do(req)
	if err != nil {
		return errors.WithStack(err)
	}

	defer res.Body.Close()

	if res.StatusCode < http.StatusOK || res.StatusCode >= http.StatusBadRequest {
		return errors.Errorf("unexpected response code %d (%s)", res.StatusCode, res.Status)
	}

	if _, err := io.Copy(result, res.Body); err != nil {
		return errors.WithStack(err)
	}

	return nil
}

func (c *TextClient) jsonRequest(ctx context.Context, method string, path string, header http.Header, body io.Reader, result any) error {
	var buff bytes.Buffer

	if header == nil {
		header = http.Header{}
	}

	header.Set("Accept", "application/json")

	if err := c.request(ctx, method, path, header, body, &buff); err != nil {
		return errors.WithStack(err)
	}

	if err := json.Unmarshal(buff.Bytes(), result); err != nil {
		return errors.WithStack(err)
	}

	return nil
}

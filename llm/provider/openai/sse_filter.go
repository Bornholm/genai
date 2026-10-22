package openai

import (
	"bufio"
	"bytes"
	"io"
	"mime"
	"net/http"

	"github.com/openai/openai-go/option"
)

// skipEmptySSEEvents is an SDK middleware that drops, from an event stream,
// the events carrying no field: comment lines such as ": PROCESSING" and the
// blank line that ends them.
//
// The SSE specification tells a client to ignore such an event, but the SDK's
// decoder dispatches every blank line and then fails to unmarshal its empty
// data with "unexpected end of JSON input". OpenAI-compatible gateways send
// these comments as keep-alives while a model takes long to produce its first
// token, so the stream would be cut on exactly the slowest calls.
func skipEmptySSEEvents(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	res, err := next(req)
	if err != nil || res == nil || res.Body == nil || !isEventStream(res.Header) {
		return res, err
	}
	res.Body = newSSEFilter(res.Body)
	return res, nil
}

func isEventStream(header http.Header) bool {
	mediaType, _, err := mime.ParseMediaType(header.Get("Content-Type"))
	return err == nil && mediaType == "text/event-stream"
}

// sseFilter forwards an event stream line by line, leaving out comment lines
// and any blank line that would end an event with no field.
type sseFilter struct {
	src    io.ReadCloser
	reader *bufio.Reader
	// pending is what was filtered but not read yet.
	pending []byte
	// inEvent records that a field was forwarded since the last blank line,
	// which makes the next blank line the end of a real event.
	inEvent bool
	err     error
}

func newSSEFilter(src io.ReadCloser) *sseFilter {
	return &sseFilter{src: src, reader: bufio.NewReader(src)}
}

func (f *sseFilter) Read(p []byte) (int, error) {
	for len(f.pending) == 0 {
		if f.err != nil {
			return 0, f.err
		}
		line, err := f.reader.ReadBytes('\n')
		f.err = err
		f.filter(line)
	}
	n := copy(p, f.pending)
	f.pending = f.pending[n:]
	return n, nil
}

func (f *sseFilter) filter(line []byte) {
	if len(line) == 0 {
		return
	}
	content := bytes.TrimRight(line, "\r\n")
	switch {
	case len(content) == 0:
		if !f.inEvent {
			return
		}
		f.inEvent = false
	case content[0] == ':':
		return
	default:
		f.inEvent = true
	}
	f.pending = append(f.pending, line...)
}

func (f *sseFilter) Close() error {
	return f.src.Close()
}

package openai

import (
	"bytes"
	"io"
	"mime"
	"net/http"

	"github.com/bornholm/genai/llm/provider/internal/sse"
	"github.com/openai/openai-go/option"
)

// skipEmptySSEEvents is an SDK middleware that keeps, in an event stream,
// only the events carrying data: comment lines such as ": PROCESSING", and
// events made only of an event name, an id or a retry field, are dropped.
//
// The SSE specification tells a client not to dispatch such an event, but the
// SDK's decoder dispatches every blank line and then fails to unmarshal the
// empty data with "unexpected end of JSON input". OpenAI-compatible gateways
// send these as keep-alives while a model takes long to produce its first
// token, so the stream would be cut on exactly the slowest calls.
//
// Every SDK response served as an event stream goes through the filter, not
// only chat completions: it only removes what the decoder cannot read anyway.
func skipEmptySSEEvents(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	res, err := next(req)
	if err != nil || res == nil || res.Body == nil || !isEventStream(res.Header) {
		return res, err
	}
	res.Body = newSSEFilter(res.Body)
	// The filtered body is shorter than the one announced.
	res.ContentLength = -1
	res.Header.Del("Content-Length")
	return res, nil
}

func isEventStream(header http.Header) bool {
	mediaType, _, err := mime.ParseMediaType(header.Get("Content-Type"))
	return err == nil && mediaType == "text/event-stream"
}

// sseFilter reads the events of a stream with the specification's rules and
// writes back those that carry data, one at a time, in the form the SDK's
// decoder expects. Like any response body, it is read from one goroutine.
type sseFilter struct {
	src    io.ReadCloser
	reader *sse.Reader
	// pending is the part of the current event not read yet.
	pending []byte
	err     error
}

func newSSEFilter(src io.ReadCloser) *sseFilter {
	return &sseFilter{src: src, reader: sse.NewReader(src)}
}

func (f *sseFilter) Read(p []byte) (int, error) {
	if len(f.pending) == 0 {
		if f.err != nil {
			return 0, f.err
		}
		event, err := f.reader.Next()
		if err != nil {
			f.err = err
			return 0, err
		}
		f.pending = encodeEvent(event)
	}
	n := copy(p, f.pending)
	f.pending = f.pending[n:]
	return n, nil
}

func (f *sseFilter) Close() error {
	return f.src.Close()
}

// encodeEvent writes event back as an event stream block: its name, then one
// data line per line of its data, then the blank line that dispatches it.
func encodeEvent(event sse.Event) []byte {
	var buf bytes.Buffer
	if event.Name != "" {
		buf.WriteString("event: ")
		buf.WriteString(event.Name)
		buf.WriteByte('\n')
	}
	for _, line := range bytes.Split(event.Data, []byte("\n")) {
		buf.WriteString("data: ")
		buf.Write(line)
		buf.WriteByte('\n')
	}
	buf.WriteByte('\n')
	return buf.Bytes()
}

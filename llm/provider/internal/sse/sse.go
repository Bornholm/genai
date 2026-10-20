// Package sse reads server-sent event streams, for the providers that speak
// their wire format by hand rather than through an SDK.
package sse

import (
	"bufio"
	"bytes"
	"io"
)

// maxLineSize caps one line of the stream. A single event carrying a whole
// response (a reasoning item with its encrypted content, a completed
// response) runs well past bufio's 64 KiB default.
const maxLineSize = 16 << 20

// Event is one server-sent event. Data joins the event's data lines with a
// newline, as the specification does.
type Event struct {
	Name string
	Data []byte
}

// Reader reads the events of a stream one at a time.
type Reader struct {
	scanner *bufio.Scanner
}

func NewReader(r io.Reader) *Reader {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64<<10), maxLineSize)
	return &Reader{scanner: scanner}
}

// Next returns the next event that carries data. Comments, retry and id
// fields are skipped. At the end of the stream it returns io.EOF; an event
// left unterminated by the end of the stream is dropped, as the
// specification requires: it is what a connection cut mid-event leaves.
func (r *Reader) Next() (Event, error) {
	var (
		event   Event
		hasData bool
	)
	for r.scanner.Scan() {
		line := bytes.TrimSuffix(r.scanner.Bytes(), []byte("\r"))
		if len(line) == 0 {
			if hasData {
				return event, nil
			}
			event = Event{}
			continue
		}
		field, value, _ := bytes.Cut(line, []byte(":"))
		value = bytes.TrimPrefix(value, []byte(" "))
		switch string(field) {
		case "event":
			event.Name = string(value)
		case "data":
			if hasData {
				event.Data = append(event.Data, '\n')
			}
			event.Data = append(event.Data, value...)
			hasData = true
		}
	}
	if err := r.scanner.Err(); err != nil {
		return Event{}, err
	}
	return Event{}, io.EOF
}

package sse

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func TestReader_ReadsEvents(t *testing.T) {
	stream := ": keep-alive\n\n" +
		"event: first\r\ndata: {\"a\":1}\r\n\r\n" +
		"data: line one\ndata: line two\nid: 7\n\n" +
		"\n\n" +
		"event: cut\ndata: never terminated"

	r := NewReader(strings.NewReader(stream))

	first, err := r.Next()
	if err != nil || first.Name != "first" || string(first.Data) != `{"a":1}` {
		t.Fatalf("first = %+v, %v", first, err)
	}

	second, err := r.Next()
	if err != nil || second.Name != "" || string(second.Data) != "line one\nline two" {
		t.Fatalf("second = %+v, %v", second, err)
	}

	if _, err := r.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("an unterminated event must be dropped, got %v", err)
	}
}

func TestReader_ReadsLongLines(t *testing.T) {
	long := strings.Repeat("x", 1<<20)
	r := NewReader(strings.NewReader("data: " + long + "\n\n"))

	event, err := r.Next()
	if err != nil || len(event.Data) != len(long) {
		t.Fatalf("got %d bytes, %v", len(event.Data), err)
	}
}

func TestReader_SkipsEventsWithEmptyData(t *testing.T) {
	r := NewReader(strings.NewReader("data:\n\nevent: ping\ndata: \n\nid: 1\n\ndata: {}\n\n"))
	event, err := r.Next()
	if err != nil || string(event.Data) != "{}" || event.Name != "" {
		t.Fatalf("event = %+v, %v", event, err)
	}
}

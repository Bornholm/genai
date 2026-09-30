package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go/shared"
	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// maxRelayErrorBody caps how much of a rejected request's body is kept.
const maxRelayErrorBody = 1 << 20

type relayTarget struct {
	baseURL string // normalized, with a trailing slash
	apiKey  string
}

// SupportsMessagesRelay reports whether the client knows its upstream, see
// llm.SupportsMessagesRelay.
func (c *ChatCompletionClient) SupportsMessagesRelay() bool {
	return c.relay != nil
}

// RelayMessages implements llm.MessagesRelayClient.
func (c *ChatCompletionClient) RelayMessages(ctx context.Context, body []byte, header http.Header) (<-chan llm.StreamChunk, error) {
	if c.relay == nil {
		return nil, errors.WithStack(llm.ErrUnavailable)
	}

	if c.model != "" {
		var err error
		if body, err = withModel(body, c.model); err != nil {
			return nil, errors.WithStack(err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.relay.baseURL+"v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, errors.WithStack(err)
	}
	for name, values := range header {
		if strings.HasPrefix(strings.ToLower(name), "anthropic-") {
			req.Header[name] = values
		}
	}
	if req.Header.Get("anthropic-version") == "" {
		req.Header.Set("anthropic-version", "2023-06-01")
	}
	req.Header.Set("x-api-key", c.relay.apiKey)
	req.Header.Set("content-type", "application/json")
	req.Header.Set("accept", "text/event-stream")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if res.StatusCode < 200 || res.StatusCode > 299 {
		defer res.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(res.Body, maxRelayErrorBody))
		err := llm.RateLimitError(res.StatusCode, string(raw))
		var httpErr *llm.HTTPError
		if errors.As(err, &httpErr) {
			httpErr.Header = res.Header
		}
		return nil, errors.WithStack(err)
	}

	if !strings.HasPrefix(res.Header.Get("content-type"), "text/event-stream") {
		res.Body.Close()
		return nil, errors.Errorf("relay: upstream answered %q instead of a stream", res.Header.Get("content-type"))
	}

	chunks := make(chan llm.StreamChunk, 10)
	go func() {
		defer close(chunks)
		defer res.Body.Close()
		relayEvents(ctx, res.Body, chunks)
		// Reading what follows message_stop, normally nothing, lets the
		// transport reuse the connection instead of dropping it.
		_, _ = io.Copy(io.Discard, io.LimitReader(res.Body, maxRelayErrorBody))
	}()

	return chunks, nil
}

// relayEvents hands every event of stream to chunks, verbatim, until the
// message_stop event or the end of the stream. A stream that ends without
// message_stop is left for the consumer to call truncated.
func relayEvents(ctx context.Context, stream io.Reader, chunks chan<- llm.StreamChunk) {
	reader := bufio.NewReader(stream)
	var usage relayUsageTracker

	for {
		event, err := readEvent(reader)
		if len(event) > 0 {
			parsed := parseEventData(event)
			usage.record(parsed)

			switch parsed.Type {
			case "message_stop":
				llm.SendTerminalChunk(ctx, chunks, llm.NewRawEventChunk(event, usage.usage(), true))
				return
			case "error":
				status := statusForErrorType(shared.ErrorType(parsed.Error.Type))
				upstreamErr := llm.RateLimitError(status, string(parsed.data))
				llm.SendTerminalChunk(ctx, chunks, llm.NewRawEventErrorChunk(event, upstreamErr, usage.usage()))
				return
			default:
				if !llm.SendChunk(ctx, chunks, llm.NewRawEventChunk(event, usage.usage(), false)) {
					llm.SendTerminalChunk(ctx, chunks, llm.NewErrorStreamChunk(errors.WithStack(ctx.Err())))
					return
				}
			}
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				llm.SendTerminalChunk(ctx, chunks, llm.NewErrorStreamChunkWithUsage(errors.WithStack(err), usage.usage()))
			}
			return
		}
	}
}

// readEvent reads one server-sent event, up to and including the blank line
// that ends it. At the end of the stream it returns what was left, which may
// be an unterminated event, along with the error.
func readEvent(reader *bufio.Reader) ([]byte, error) {
	var event []byte
	for {
		line, err := reader.ReadBytes('\n')
		event = append(event, line...)
		if err != nil {
			return event, err
		}
		if len(bytes.TrimRight(line, "\r\n")) == 0 {
			if len(bytes.TrimSpace(event)) == 0 {
				// Stray blank lines between events carry nothing.
				event = event[:0]
				continue
			}
			return event, nil
		}
	}
}

type relayEventUsage struct {
	InputTokens              *int64 `json:"input_tokens"`
	OutputTokens             *int64 `json:"output_tokens"`
	CacheReadInputTokens     *int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens *int64 `json:"cache_creation_input_tokens"`
}

type relayEventData struct {
	Type    string `json:"type"`
	Message struct {
		Usage relayEventUsage `json:"usage"`
	} `json:"message"`
	Usage relayEventUsage `json:"usage"`
	Error struct {
		Type string `json:"type"`
	} `json:"error"`

	data []byte
}

// parseEventData decodes the data line of event. Events the relay does not
// need to understand, or cannot, decode to an empty type and pass through.
func parseEventData(event []byte) relayEventData {
	var parsed relayEventData
	for _, line := range bytes.Split(event, []byte("\n")) {
		if data, ok := bytes.CutPrefix(bytes.TrimRight(line, "\r"), []byte("data:")); ok {
			parsed.data = bytes.TrimSpace(data)
			_ = json.Unmarshal(parsed.data, &parsed)
			break
		}
	}
	return parsed
}

// relayUsageTracker accumulates the counters the way the API publishes them:
// message_start carries the input side, message_delta the running output
// count and, on some models, updated input counts.
type relayUsageTracker struct {
	input, output, cacheRead, cacheCreation int64
	seen                                    bool
}

func (t *relayUsageTracker) record(event relayEventData) {
	switch event.Type {
	case "message_start":
		t.apply(event.Message.Usage)
	case "message_delta":
		t.apply(event.Usage)
	}
}

func (t *relayUsageTracker) apply(u relayEventUsage) {
	if u.InputTokens != nil {
		t.input, t.seen = *u.InputTokens, true
	}
	if u.OutputTokens != nil {
		t.output, t.seen = *u.OutputTokens, true
	}
	if u.CacheReadInputTokens != nil {
		t.cacheRead, t.seen = *u.CacheReadInputTokens, true
	}
	if u.CacheCreationInputTokens != nil {
		t.cacheCreation, t.seen = *u.CacheCreationInputTokens, true
	}
}

func (t *relayUsageTracker) usage() llm.ChatCompletionUsage {
	if !t.seen {
		return nil
	}
	return newUsage(t.input, t.output, t.cacheRead, t.cacheCreation)
}

// withModel sets the model of a request body. The other fields keep their
// values, not their bytes: the body is re-encoded, keys sorted and whitespace
// compacted, which the API reads the same.
func withModel(body []byte, model string) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, errors.Wrap(err, "relay: request body is not a JSON object")
	}
	encoded, err := json.Marshal(model)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	fields["model"] = encoded

	// Without SetEscapeHTML(false) every <, > and & of the conversation would
	// be rewritten as \u003c and friends: the same JSON, but not the bytes the
	// client sent.
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(fields); err != nil {
		return nil, errors.WithStack(err)
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

var _ llm.MessagesRelayClient = &ChatCompletionClient{}

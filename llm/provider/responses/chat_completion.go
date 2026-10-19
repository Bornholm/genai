package responses

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/llm/provider/internal/sse"
	"github.com/pkg/errors"
)

// maxErrorBody caps how much of a rejected request's body is kept.
const maxErrorBody = 1 << 20

// ChatCompletionClient talks to the OpenAI Responses API (POST /responses).
type ChatCompletionClient struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
	model      string
}

// ChatCompletion implements llm.ChatCompletionClient.
func (c *ChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	req, err := buildRequest(opts, c.model, false)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	res, err := c.post(ctx, req)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	defer res.Body.Close()

	var body response
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		return nil, errors.WithStack(llm.RateLimitError(http.StatusBadGateway, "malformed response: "+err.Error()))
	}

	result, err := fromResponse(&body, excludeReasoning(opts))
	if err != nil {
		return nil, errors.WithStack(err)
	}
	return result, nil
}

// ChatCompletionStream implements llm.ChatCompletionStreamingClient.
func (c *ChatCompletionClient) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	req, err := buildRequest(opts, c.model, true)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	res, err := c.post(ctx, req)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if contentType := res.Header.Get("Content-Type"); !strings.HasPrefix(contentType, "text/event-stream") {
		res.Body.Close()
		return nil, errors.WithStack(llm.RateLimitError(http.StatusBadGateway, "upstream answered "+contentType+" instead of an event stream"))
	}

	chunks := make(chan llm.StreamChunk, 10)

	go func() {
		defer close(chunks)
		defer res.Body.Close()

		emitter := newStreamEmitter(ctx, chunks, excludeReasoning(opts))
		reader := sse.NewReader(res.Body)

		for {
			event, err := reader.Next()
			if err != nil {
				if errors.Is(err, io.EOF) {
					// The API always ends a stream with a terminal event:
					// without one, the stream was cut.
					emitter.fail(errStreamTruncated())
				} else {
					emitter.fail(errors.WithStack(err))
				}
				return
			}
			if done := emitter.handle(event.Data); done {
				return
			}
		}
	}()

	return chunks, nil
}

// post sends req and returns the response when its status is 2xx, an error
// wrapping *llm.HTTPError otherwise.
func (c *ChatCompletionClient) post(ctx context.Context, req *request) (*http.Response, error) {
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/responses", bytes.NewReader(payload))
	if err != nil {
		return nil, errors.WithStack(err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if req.Stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	} else {
		httpReq.Header.Set("Accept", "application/json")
	}
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	res, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if res.StatusCode < 200 || res.StatusCode > 299 {
		defer res.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(res.Body, maxErrorBody))
		err := llm.RateLimitError(res.StatusCode, string(raw))
		var httpErr *llm.HTTPError
		if errors.As(err, &httpErr) {
			httpErr.Header = res.Header
		}
		return nil, errors.WithStack(err)
	}

	return res, nil
}

// excludeReasoning reports whether the caller asked not to see the
// reasoning. The encrypted reasoning is still carried in the details: the
// next turn replays it.
func excludeReasoning(opts *llm.ChatCompletionOptions) bool {
	return opts.Reasoning != nil && opts.Reasoning.Exclude
}

// errStreamTruncated reports a stream that ended without a terminal event.
// It is a 502, the status a gateway gives to a cut upstream, so that the call
// is retried when nothing was forwarded yet. Each call builds its own: an
// *llm.HTTPError is mutable.
func errStreamTruncated() error {
	return errors.WithStack(llm.NewHTTPError(http.StatusBadGateway, "responses: the stream ended before the response completed"))
}

// NewChatCompletionClient returns a client posting to baseURL, the API root
// (".../v1"). A nil httpClient uses http.DefaultClient.
func NewChatCompletionClient(httpClient *http.Client, baseURL, apiKey, model string) *ChatCompletionClient {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &ChatCompletionClient{
		httpClient: httpClient,
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		model:      model,
	}
}

var _ llm.ChatCompletionClient = &ChatCompletionClient{}
var _ llm.ChatCompletionStreamingClient = &ChatCompletionClient{}

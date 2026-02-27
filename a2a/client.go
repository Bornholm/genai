package a2a

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/pkg/errors"
)

// Client is an A2A protocol client
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a new A2A client
func NewClient(baseURL string, funcs ...ClientOptionFunc) *Client {
	opts := NewClientOptions(funcs...)
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: opts.HTTPClient,
	}
}

// GetAgentCard fetches the agent card from /.well-known/agent.json
func (c *Client) GetAgentCard(ctx context.Context) (*AgentCard, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/.well-known/agent.json", nil)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, errors.Errorf("unexpected status %d fetching agent card", resp.StatusCode)
	}

	var card AgentCard
	if err := json.NewDecoder(resp.Body).Decode(&card); err != nil {
		return nil, errors.WithStack(err)
	}
	return &card, nil
}

// SendTask sends a tasks/send request and returns the completed task.
func (c *Client) SendTask(ctx context.Context, params TaskSendParams) (*Task, error) {
	result, err := c.call(ctx, MethodTasksSend, params)
	if err != nil {
		return nil, err
	}

	var task Task
	if err := json.Unmarshal(result, &task); err != nil {
		return nil, errors.WithStack(err)
	}
	return &task, nil
}

// StreamEvent represents a single event from the SSE stream
type StreamEvent struct {
	Data json.RawMessage
	Err  error
}

// SendTaskSubscribe sends a tasks/sendSubscribe request and streams events.
func (c *Client) SendTaskSubscribe(ctx context.Context, params TaskSendParams) (<-chan StreamEvent, error) {
	body, err := json.Marshal(JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  MethodTasksSendSubscribe,
		Params:  mustMarshalRaw(params),
	})
	if err != nil {
		return nil, errors.WithStack(err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/", bytes.NewReader(body))
	if err != nil {
		return nil, errors.WithStack(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	ch := make(chan StreamEvent, 32)

	go func() {
		defer close(ch)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")

			var rpcResp JSONRPCResponse
			if err := json.Unmarshal([]byte(data), &rpcResp); err != nil {
				ch <- StreamEvent{Err: err}
				return
			}

			if rpcResp.Error != nil {
				ch <- StreamEvent{Err: fmt.Errorf("rpc error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)}
				return
			}

			resultBytes, _ := json.Marshal(rpcResp.Result)
			ch <- StreamEvent{Data: resultBytes}
		}
	}()

	return ch, nil
}

// GetTask retrieves a task.
func (c *Client) GetTask(ctx context.Context, params TaskQueryParams) (*Task, error) {
	result, err := c.call(ctx, MethodTasksGet, params)
	if err != nil {
		return nil, err
	}

	var task Task
	if err := json.Unmarshal(result, &task); err != nil {
		return nil, errors.WithStack(err)
	}
	return &task, nil
}

// CancelTask cancels a task.
func (c *Client) CancelTask(ctx context.Context, params TaskQueryParams) (*Task, error) {
	result, err := c.call(ctx, MethodTasksCancel, params)
	if err != nil {
		return nil, err
	}

	var task Task
	if err := json.Unmarshal(result, &task); err != nil {
		return nil, errors.WithStack(err)
	}
	return &task, nil
}

func (c *Client) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	body, err := json.Marshal(JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  method,
		Params:  mustMarshalRaw(params),
	})
	if err != nil {
		return nil, errors.WithStack(err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/", bytes.NewReader(body))
	if err != nil {
		return nil, errors.WithStack(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	defer resp.Body.Close()

	// Read the body for better error messages
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	var rpcResp JSONRPCResponse
	if err := json.Unmarshal(bodyBytes, &rpcResp); err != nil {
		return nil, errors.Wrapf(err, "failed to parse response: %s", string(bodyBytes))
	}

	if rpcResp.Error != nil {
		return nil, fmt.Errorf("rpc error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}

	resultBytes, err := json.Marshal(rpcResp.Result)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return resultBytes, nil
}

func mustMarshalRaw(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

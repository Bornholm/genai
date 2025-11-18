package circuitbreaker

import (
	"context"
	"sync"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// State represents the circuit breaker state
type State int

const (
	StateClosed State = iota
	StateOpen
	StateHalfOpen
)

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
	maxFailures  int
	resetTimeout time.Duration
	failures     int
	lastFailTime time.Time
	state        State
	mutex        sync.RWMutex
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		maxFailures:  maxFailures,
		resetTimeout: resetTimeout,
		state:        StateClosed,
	}
}

// Execute runs the function with circuit breaker protection
func (cb *CircuitBreaker) Execute(fn func() error) error {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	if cb.state == StateOpen {
		if time.Since(cb.lastFailTime) > cb.resetTimeout {
			cb.state = StateHalfOpen
		} else {
			return errors.New("circuit breaker is open")
		}
	}

	err := fn()
	if err != nil {
		cb.failures++
		cb.lastFailTime = time.Now()

		if cb.failures >= cb.maxFailures {
			cb.state = StateOpen
		}
		return err
	}

	// Success - reset failures and close circuit
	cb.failures = 0
	cb.state = StateClosed
	return nil
}

// Client wraps an LLM client with circuit breaker protection
type Client struct {
	client  llm.Client
	breaker *CircuitBreaker
}

// NewClient creates a new circuit breaker protected client
func NewClient(client llm.Client, maxFailures int, resetTimeout time.Duration) *Client {
	return &Client{
		client:  client,
		breaker: NewCircuitBreaker(maxFailures, resetTimeout),
	}
}

// ChatCompletion implements llm.Client with circuit breaker protection
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	var response llm.ChatCompletionResponse
	var err error

	breakerErr := c.breaker.Execute(func() error {
		response, err = c.client.ChatCompletion(ctx, funcs...)
		return err
	})

	if breakerErr != nil {
		return nil, errors.WithStack(breakerErr)
	}

	return response, err
}

// Embeddings implements llm.Client with circuit breaker protection
func (c *Client) Embeddings(ctx context.Context, input string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	var response llm.EmbeddingsResponse
	var err error

	breakerErr := c.breaker.Execute(func() error {
		response, err = c.client.Embeddings(ctx, input, funcs...)
		return err
	})

	if breakerErr != nil {
		return nil, errors.WithStack(breakerErr)
	}

	return response, err
}

var _ llm.Client = &Client{}

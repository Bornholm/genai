package hook

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type BeforeChatCompletionHook interface {
	BeforeChatCompletion(ctx context.Context, funcs []llm.ChatCompletionOptionFunc) (context.Context, []llm.ChatCompletionOptionFunc, error)
}

type BeforeChatCompletionFunc func(ctx context.Context, funcs []llm.ChatCompletionOptionFunc) (context.Context, []llm.ChatCompletionOptionFunc, error)

func (fn BeforeChatCompletionFunc) BeforeChatCompletion(ctx context.Context, funcs []llm.ChatCompletionOptionFunc) (context.Context, []llm.ChatCompletionOptionFunc, error) {
	return fn(ctx, funcs)
}

type AfterChatCompletionHook interface {
	AfterChatCompletion(ctx context.Context, funcs []llm.ChatCompletionOptionFunc, res llm.ChatCompletionResponse) (llm.ChatCompletionResponse, error)
}

type AfterChatCompletionFunc func(ctx context.Context, funcs []llm.ChatCompletionOptionFunc, res llm.ChatCompletionResponse) (llm.ChatCompletionResponse, error)

func (fn AfterChatCompletionFunc) AfterChatCompletion(ctx context.Context, funcs []llm.ChatCompletionOptionFunc, res llm.ChatCompletionResponse) (llm.ChatCompletionResponse, error) {
	return fn(ctx, funcs, res)
}

type BeforeChatCompletionStreamHook interface {
	BeforeChatCompletionStream(ctx context.Context, funcs []llm.ChatCompletionOptionFunc) (context.Context, []llm.ChatCompletionOptionFunc, error)
}

type BeforeChatCompletionStreamFunc func(ctx context.Context, funcs []llm.ChatCompletionOptionFunc) (context.Context, []llm.ChatCompletionOptionFunc, error)

func (fn BeforeChatCompletionStreamFunc) BeforeChatCompletionStream(ctx context.Context, funcs []llm.ChatCompletionOptionFunc) (context.Context, []llm.ChatCompletionOptionFunc, error) {
	return fn(ctx, funcs)
}

type AfterChatCompletionStreamHook interface {
	AfterChatCompletionStream(ctx context.Context, funcs []llm.ChatCompletionOptionFunc, stream <-chan llm.StreamChunk) (<-chan llm.StreamChunk, error)
}

type AfterChatCompletionStreamFunc func(ctx context.Context, funcs []llm.ChatCompletionOptionFunc, stream <-chan llm.StreamChunk) (<-chan llm.StreamChunk, error)

func (fn AfterChatCompletionStreamFunc) AfterChatCompletionStream(ctx context.Context, funcs []llm.ChatCompletionOptionFunc, stream <-chan llm.StreamChunk) (<-chan llm.StreamChunk, error) {
	return fn(ctx, funcs, stream)
}

type EmbeddingsHook interface {
	BeforeEmbeddings(ctx context.Context, inputs []string, funcs []llm.EmbeddingsOptionFunc) (context.Context, []string, []llm.EmbeddingsOptionFunc, error)
	AfterEmbeddings(ctx context.Context, inputs []string, funcs []llm.EmbeddingsOptionFunc, res llm.EmbeddingsResponse) (llm.EmbeddingsResponse, error)
}

type BeforeEmbeddingsHook interface {
	BeforeEmbeddings(ctx context.Context, inputs []string, funcs []llm.EmbeddingsOptionFunc) (context.Context, []string, []llm.EmbeddingsOptionFunc, error)
}

type BeforeEmbeddingsFunc func(ctx context.Context, inputs []string, funcs []llm.EmbeddingsOptionFunc) (context.Context, []string, []llm.EmbeddingsOptionFunc, error)

func (fn BeforeEmbeddingsFunc) BeforeEmbeddings(ctx context.Context, inputs []string, funcs []llm.EmbeddingsOptionFunc) (context.Context, []string, []llm.EmbeddingsOptionFunc, error) {
	return fn(ctx, inputs, funcs)
}

type AfterEmbeddingsHook interface {
	AfterEmbeddings(ctx context.Context, inputs []string, funcs []llm.EmbeddingsOptionFunc, res llm.EmbeddingsResponse) (llm.EmbeddingsResponse, error)
}

type AfterEmbeddingsFunc func(ctx context.Context, inputs []string, funcs []llm.EmbeddingsOptionFunc, res llm.EmbeddingsResponse) (llm.EmbeddingsResponse, error)

func (fn AfterEmbeddingsFunc) AfterEmbeddings(ctx context.Context, inputs []string, funcs []llm.EmbeddingsOptionFunc, res llm.EmbeddingsResponse) (llm.EmbeddingsResponse, error) {
	return fn(ctx, inputs, funcs, res)
}

type BeforeTranscriptionHook interface {
	BeforeTranscription(ctx context.Context, audio []byte, funcs []llm.TranscriptionOptionFunc) (context.Context, []byte, []llm.TranscriptionOptionFunc, error)
}

type BeforeTranscriptionFunc func(ctx context.Context, audio []byte, funcs []llm.TranscriptionOptionFunc) (context.Context, []byte, []llm.TranscriptionOptionFunc, error)

func (fn BeforeTranscriptionFunc) BeforeTranscription(ctx context.Context, audio []byte, funcs []llm.TranscriptionOptionFunc) (context.Context, []byte, []llm.TranscriptionOptionFunc, error) {
	return fn(ctx, audio, funcs)
}

type AfterTranscriptionHook interface {
	AfterTranscription(ctx context.Context, audio []byte, funcs []llm.TranscriptionOptionFunc, res llm.TranscriptionResponse) (llm.TranscriptionResponse, error)
}

type AfterTranscriptionFunc func(ctx context.Context, audio []byte, funcs []llm.TranscriptionOptionFunc, res llm.TranscriptionResponse) (llm.TranscriptionResponse, error)

func (fn AfterTranscriptionFunc) AfterTranscription(ctx context.Context, audio []byte, funcs []llm.TranscriptionOptionFunc, res llm.TranscriptionResponse) (llm.TranscriptionResponse, error) {
	return fn(ctx, audio, funcs, res)
}

type Client struct {
	client llm.Client

	beforeChatCompletion       BeforeChatCompletionHook
	afterChatCompletion        AfterChatCompletionHook
	beforeChatCompletionStream BeforeChatCompletionStreamHook
	afterChatCompletionStream  AfterChatCompletionStreamHook

	beforeEmbeddings BeforeEmbeddingsHook
	afterEmbeddings  AfterEmbeddingsHook

	beforeTranscription BeforeTranscriptionHook
	afterTranscription  AfterTranscriptionHook
}

type Options struct {
	BeforeChatCompletion       BeforeChatCompletionHook
	AfterChatCompletion        AfterChatCompletionHook
	BeforeChatCompletionStream BeforeChatCompletionStreamHook
	AfterChatCompletionStream  AfterChatCompletionStreamHook
	BeforeEmbeddings           BeforeEmbeddingsHook
	AfterEmbeddings            AfterEmbeddingsHook
	BeforeTranscription        BeforeTranscriptionHook
	AfterTranscription         AfterTranscriptionHook
}

type OptionFunc func(opts *Options)

func NewOptions(funcs ...OptionFunc) *Options {
	opts := &Options{}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

func WithBeforeChatCompletion(hook BeforeChatCompletionHook) OptionFunc {
	return func(opts *Options) {
		opts.BeforeChatCompletion = hook
	}
}

func WithBeforeChatCompletionFunc(fn BeforeChatCompletionFunc) OptionFunc {
	return WithBeforeChatCompletion(BeforeChatCompletionHook(fn))
}

func WithAfterChatCompletion(hook AfterChatCompletionHook) OptionFunc {
	return func(opts *Options) {
		opts.AfterChatCompletion = hook
	}
}

func WithAfterChatCompletionFunc(fn AfterChatCompletionFunc) OptionFunc {
	return WithAfterChatCompletion(AfterChatCompletionHook(fn))
}

func WithBeforeEmbeddings(hook BeforeEmbeddingsHook) OptionFunc {
	return func(opts *Options) {
		opts.BeforeEmbeddings = hook
	}
}

func WithBeforeEmbeddingsFunc(fn BeforeEmbeddingsFunc) OptionFunc {
	return WithBeforeEmbeddings(BeforeEmbeddingsHook(fn))
}

func WithAfterEmbeddings(hook AfterEmbeddingsHook) OptionFunc {
	return func(opts *Options) {
		opts.AfterEmbeddings = hook
	}
}

func WithAfterEmbeddingsFunc(fn AfterEmbeddingsFunc) OptionFunc {
	return WithAfterEmbeddings(AfterEmbeddingsHook(fn))
}

func WithBeforeTranscription(hook BeforeTranscriptionHook) OptionFunc {
	return func(opts *Options) {
		opts.BeforeTranscription = hook
	}
}

func WithBeforeTranscriptionFunc(fn BeforeTranscriptionFunc) OptionFunc {
	return WithBeforeTranscription(BeforeTranscriptionHook(fn))
}

func WithAfterTranscription(hook AfterTranscriptionHook) OptionFunc {
	return func(opts *Options) {
		opts.AfterTranscription = hook
	}
}

func WithAfterTranscriptionFunc(fn AfterTranscriptionFunc) OptionFunc {
	return WithAfterTranscription(AfterTranscriptionHook(fn))
}

func WithBeforeChatCompletionStream(hook BeforeChatCompletionStreamHook) OptionFunc {
	return func(opts *Options) {
		opts.BeforeChatCompletionStream = hook
	}
}

func WithBeforeChatCompletionStreamFunc(fn BeforeChatCompletionStreamFunc) OptionFunc {
	return WithBeforeChatCompletionStream(BeforeChatCompletionStreamHook(fn))
}

func WithAfterChatCompletionStream(hook AfterChatCompletionStreamHook) OptionFunc {
	return func(opts *Options) {
		opts.AfterChatCompletionStream = hook
	}
}

func WithAfterChatCompletionStreamFunc(fn AfterChatCompletionStreamFunc) OptionFunc {
	return WithAfterChatCompletionStream(AfterChatCompletionStreamHook(fn))
}

// ChatCompletion implements llm.Client.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	var err error

	if c.beforeChatCompletion != nil {
		ctx, funcs, err = c.beforeChatCompletion.BeforeChatCompletion(ctx, funcs)
		if err != nil {
			return nil, errors.WithStack(err)
		}
	}

	res, err := c.client.ChatCompletion(ctx, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if c.afterChatCompletion != nil {
		res, err = c.afterChatCompletion.AfterChatCompletion(ctx, funcs, res)
		if err != nil {
			return nil, errors.WithStack(err)
		}
	}

	return res, nil
}

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	var err error

	if c.beforeEmbeddings != nil {
		ctx, inputs, funcs, err = c.beforeEmbeddings.BeforeEmbeddings(ctx, inputs, funcs)
		if err != nil {
			return nil, errors.WithStack(err)
		}
	}

	res, err := c.client.Embeddings(ctx, inputs, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if c.afterEmbeddings != nil {
		res, err = c.afterEmbeddings.AfterEmbeddings(ctx, inputs, funcs, res)
		if err != nil {
			return nil, errors.WithStack(err)
		}
	}

	return res, nil
}

// Transcription implements llm.Client.
func (c *Client) Transcription(ctx context.Context, audio []byte, funcs ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	var err error

	if c.beforeTranscription != nil {
		ctx, audio, funcs, err = c.beforeTranscription.BeforeTranscription(ctx, audio, funcs)
		if err != nil {
			return nil, errors.WithStack(err)
		}
	}

	res, err := c.client.Transcription(ctx, audio, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if c.afterTranscription != nil {
		res, err = c.afterTranscription.AfterTranscription(ctx, audio, funcs, res)
		if err != nil {
			return nil, errors.WithStack(err)
		}
	}

	return res, nil
}

// ChatCompletionStream implements llm.Client.
func (c *Client) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	var err error

	if c.beforeChatCompletionStream != nil {
		ctx, funcs, err = c.beforeChatCompletionStream.BeforeChatCompletionStream(ctx, funcs)
		if err != nil {
			return nil, errors.WithStack(err)
		}
	}

	stream, err := c.client.ChatCompletionStream(ctx, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if c.afterChatCompletionStream != nil {
		stream, err = c.afterChatCompletionStream.AfterChatCompletionStream(ctx, funcs, stream)
		if err != nil {
			return nil, errors.WithStack(err)
		}
	}

	return stream, nil
}

func NewClient(client llm.Client, funcs ...OptionFunc) *Client {
	opts := NewOptions(funcs...)
	return &Client{
		client:                     client,
		beforeChatCompletion:       opts.BeforeChatCompletion,
		afterChatCompletion:        opts.AfterChatCompletion,
		beforeChatCompletionStream: opts.BeforeChatCompletionStream,
		afterChatCompletionStream:  opts.AfterChatCompletionStream,
		beforeEmbeddings:           opts.BeforeEmbeddings,
		afterEmbeddings:            opts.AfterEmbeddings,
		beforeTranscription:        opts.BeforeTranscription,
		afterTranscription:         opts.AfterTranscription,
	}
}

var _ llm.Client = &Client{}

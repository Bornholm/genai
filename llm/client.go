package llm

import (
	"context"
)

type ResponseFormat string

const (
	ResponseFormatJSON    ResponseFormat = "json"
	ResponseFormatDefault ResponseFormat = "default"
)

type ToolChoice string

const (
	ToolChoiceAuto    ToolChoice = "auto"
	ToolChoiceDefault ToolChoice = "default"
)

type ChatCompletionOptions struct {
	Messages       []Message
	Tools          []Tool
	ToolChoice     ToolChoice
	Temperature    float64
	ResponseFormat ResponseFormat
	ResponseSchema ResponseSchema
}

func NewChatCompletionOptions(funcs ...ChatCompletionOptionFunc) *ChatCompletionOptions {
	opts := &ChatCompletionOptions{
		Messages:       make([]Message, 0),
		Tools:          make([]Tool, 0),
		Temperature:    0.6,
		ResponseFormat: ResponseFormatDefault,
		ResponseSchema: nil,
		ToolChoice:     ToolChoiceDefault,
	}
	for _, fn := range funcs {
		fn(opts)
	}
	return opts
}

type ChatCompletionOptionFunc func(opts *ChatCompletionOptions)

func WithToolChoice(choice ToolChoice) ChatCompletionOptionFunc {
	return func(opts *ChatCompletionOptions) {
		opts.ToolChoice = choice
	}
}

func WithTemperature(temperature float64) ChatCompletionOptionFunc {
	return func(opts *ChatCompletionOptions) {
		opts.Temperature = temperature
	}
}

func WithResponseFormat(format ResponseFormat) ChatCompletionOptionFunc {
	return func(opts *ChatCompletionOptions) {
		opts.ResponseFormat = format
	}
}

func WithResponseSchema(schema ResponseSchema) ChatCompletionOptionFunc {
	return func(opts *ChatCompletionOptions) {
		opts.ResponseSchema = schema
	}
}

func WithMessages(messages ...Message) ChatCompletionOptionFunc {
	return func(opts *ChatCompletionOptions) {
		opts.Messages = messages
	}
}

type ResponseSchema interface {
	Name() string
	Description() string
	Schema() any
}

type BaseResponseSchema struct {
	name        string
	description string
	schema      any
}

// Description implements ResponseSchema.
func (b *BaseResponseSchema) Description() string {
	return b.description
}

// Name implements ResponseSchema.
func (b *BaseResponseSchema) Name() string {
	return b.name
}

// Schema implements ResponseSchema.
func (b *BaseResponseSchema) Schema() any {
	return b.schema
}

var _ ResponseSchema = &BaseResponseSchema{}

func NewResponseSchema(name string, description string, schema any) *BaseResponseSchema {
	return &BaseResponseSchema{
		name:        name,
		description: description,
		schema:      schema,
	}
}

func WithJSONResponse(schema ResponseSchema) ChatCompletionOptionFunc {
	return func(opts *ChatCompletionOptions) {
		opts.ResponseFormat = ResponseFormatJSON
		opts.ResponseSchema = schema
	}
}

func WithTools(tools ...Tool) ChatCompletionOptionFunc {
	return func(opts *ChatCompletionOptions) {
		if opts.Tools == nil {
			opts.Tools = make([]Tool, 0)
		}

		opts.Tools = append(opts.Tools, tools...)
	}
}

type ChatCompletionClient interface {
	ChatCompletion(ctx context.Context, funcs ...ChatCompletionOptionFunc) (CompletionResponse, error)
}

type EmbeddingsClient interface {
	Embeddings(ctx context.Context, input string, funcs ...EmbeddingsOptionFunc) (EmbeddingsResponse, error)
}

type OCRClient interface {
	OCR(ctx context.Context, funcs ...OCROptionFunc) (OCRResponse, error)
}

type Client interface {
	ChatCompletionClient
	EmbeddingsClient
}

type Role string

const (
	RoleAssistant = "assistant"
	RoleUser      = "user"
	RoleSystem    = "system"
	RoleTool      = "tool"
	RoleToolCalls = "tool_calls"
)

type Message interface {
	Role() Role
	Content() string
}

type BaseMessage struct {
	role    Role
	content string
}

// Content implements Message.
func (b *BaseMessage) Content() string {
	return b.content
}

// Role implements Message.
func (b *BaseMessage) Role() Role {
	return b.role
}

var _ Message = &BaseMessage{}

func NewMessage(role Role, content string) *BaseMessage {
	return &BaseMessage{
		role:    role,
		content: content,
	}
}

type ToolMessage interface {
	Message
	ID() string
}

type BaseToolMessage struct {
	BaseMessage
	id string
}

func (b *BaseToolMessage) ID() string {
	return b.id
}

var _ ToolMessage = &BaseToolMessage{}

func NewToolMessage(id string, content string) *BaseToolMessage {
	return &BaseToolMessage{
		BaseMessage: BaseMessage{
			role:    RoleTool,
			content: content,
		},
		id: id,
	}
}

type ToolCallsMessage interface {
	Message
	ToolCalls() []ToolCall
}

type BaseToolCallsMessage struct {
	BaseMessage
	toolCalls []ToolCall
}

func (b *BaseToolCallsMessage) ToolCalls() []ToolCall {
	return b.toolCalls
}

var _ ToolCallsMessage = &BaseToolCallsMessage{}

func NewToolCallsMessage(toolCalls ...ToolCall) *BaseToolCallsMessage {
	return &BaseToolCallsMessage{
		BaseMessage: BaseMessage{
			role:    RoleToolCalls,
			content: "",
		},
		toolCalls: toolCalls,
	}
}

type CompletionResponse interface {
	Message() Message
	ToolCalls() []ToolCall
}

type BaseCompletionResponse struct {
	message   Message
	toolCalls []ToolCall
}

// Content implements CompletionResponse.
func (b *BaseCompletionResponse) Message() Message {
	return b.message
}

// ToolCalls implements CompletionResponse.
func (b *BaseCompletionResponse) ToolCalls() []ToolCall {
	return b.toolCalls
}

func NewCompletionResponse(message Message, toolCalls ...ToolCall) *BaseCompletionResponse {
	return &BaseCompletionResponse{
		message:   message,
		toolCalls: toolCalls,
	}
}

var _ CompletionResponse = &BaseCompletionResponse{}

type ToolCall interface {
	ToolCallsMessage
	ID() string
	Name() string
	Parameters() any
}

type BaseToolCall struct {
	id         string
	name       string
	parameters any
}

// ToolCalls implements ToolCall.
func (b *BaseToolCall) ToolCalls() []ToolCall {
	return []ToolCall{b}
}

// Content implements ToolCall.
func (b *BaseToolCall) Content() string {
	return ""
}

// Role implements ToolCall.
func (b *BaseToolCall) Role() Role {
	return RoleToolCalls
}

// ID implements ToolCall.
func (b *BaseToolCall) ID() string {
	return b.id
}

// Name implements ToolCall.
func (b *BaseToolCall) Name() string {
	return b.name
}

// Parameters implements ToolCall.
func (b *BaseToolCall) Parameters() any {
	return b.parameters
}

var _ ToolCall = &BaseToolCall{}

func NewToolCall(id string, name string, parameters string) *BaseToolCall {
	return &BaseToolCall{
		id:         id,
		name:       name,
		parameters: parameters,
	}
}

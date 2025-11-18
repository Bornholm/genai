package llm

import (
	"context"
)

type ChatCompletionClient interface {
	ChatCompletion(ctx context.Context, funcs ...ChatCompletionOptionFunc) (ChatCompletionResponse, error)
}

type ResponseFormat string

const (
	ResponseFormatJSON    ResponseFormat = "json"
	ResponseFormatDefault ResponseFormat = "default"
)

type ToolChoice string

const (
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceRequired ToolChoice = "required"
)

type ChatCompletionOptions struct {
	Messages            []Message
	Tools               []Tool
	ToolChoice          ToolChoice
	Temperature         float64
	ResponseFormat      ResponseFormat
	ResponseSchema      ResponseSchema
	Seed                *int
	MaxCompletionTokens *int
}

// Validate checks if the ChatCompletionOptions are valid
func (opts *ChatCompletionOptions) Validate() error {
	if opts.Temperature < 0 || opts.Temperature > 2 {
		return NewValidationError("temperature", "temperature must be between 0 and 2")
	}
	if opts.MaxCompletionTokens != nil && *opts.MaxCompletionTokens <= 0 {
		return NewValidationError("max_completion_tokens", "max completion tokens must be positive")
	}
	if len(opts.Messages) == 0 {
		return NewValidationError("messages", "at least one message is required")
	}
	// Validate that we have at least one non-empty message
	hasContent := false
	for _, msg := range opts.Messages {
		if msg.Content() != "" {
			hasContent = true
			break
		}
	}
	if !hasContent {
		return NewValidationError("messages", "at least one message must have content")
	}
	return nil
}

func NewChatCompletionOptions(funcs ...ChatCompletionOptionFunc) *ChatCompletionOptions {
	opts := &ChatCompletionOptions{
		Messages:       make([]Message, 0),
		Tools:          make([]Tool, 0),
		Temperature:    0.6,
		ResponseFormat: ResponseFormatDefault,
		ResponseSchema: nil,
		ToolChoice:     ToolChoiceNone,
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

func WithSeed(seed int) ChatCompletionOptionFunc {
	return func(opts *ChatCompletionOptions) {
		opts.Seed = &seed
	}
}

func WithMaxCompletionTokens(maxTokens int) ChatCompletionOptionFunc {
	return func(opts *ChatCompletionOptions) {
		opts.MaxCompletionTokens = &maxTokens
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

type Role string

const (
	RoleAssistant Role = "assistant"
	RoleUser      Role = "user"
	RoleSystem    Role = "system"
	RoleTool      Role = "tool"
	RoleToolCalls Role = "tool_calls"
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

type ChatCompletionResponse interface {
	Message() Message
	ToolCalls() []ToolCall
	Usage() ChatCompletionUsage
}

type ChatCompletionUsage interface {
	TotalTokens() int64
	PromptTokens() int64
	CompletionTokens() int64
}

type BaseChatCompletionResponse struct {
	message   Message
	toolCalls []ToolCall
	usage     ChatCompletionUsage
}

// Usage implements ChatCompletionResponse.
func (b *BaseChatCompletionResponse) Usage() ChatCompletionUsage {
	return b.usage
}

// Content implements CompletionResponse.
func (b *BaseChatCompletionResponse) Message() Message {
	return b.message
}

// ToolCalls implements CompletionResponse.
func (b *BaseChatCompletionResponse) ToolCalls() []ToolCall {
	return b.toolCalls
}

func NewChatCompletionResponse(message Message, usage ChatCompletionUsage, toolCalls ...ToolCall) *BaseChatCompletionResponse {
	return &BaseChatCompletionResponse{
		usage:     usage,
		message:   message,
		toolCalls: toolCalls,
	}
}

var _ ChatCompletionResponse = &BaseChatCompletionResponse{}

type BaseChatCompletionUsage struct {
	totalTokens      int64
	promptTokens     int64
	completionTokens int64
}

// CompletionTokens implements ChatCompletionUsage.
func (u *BaseChatCompletionUsage) CompletionTokens() int64 {
	return u.completionTokens
}

// PromptTokens implements ChatCompletionUsage.
func (u *BaseChatCompletionUsage) PromptTokens() int64 {
	return u.promptTokens
}

// TotalTokens implements ChatCompletionUsage.
func (u *BaseChatCompletionUsage) TotalTokens() int64 {
	return u.totalTokens
}

func NewChatCompletionUsage(promptTokens, completionTokens, totalTokens int64) *BaseChatCompletionUsage {
	return &BaseChatCompletionUsage{
		promptTokens:     promptTokens,
		completionTokens: completionTokens,
		totalTokens:      totalTokens,
	}
}

var _ ChatCompletionUsage = &BaseChatCompletionUsage{}

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

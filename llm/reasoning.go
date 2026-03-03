package llm

// ReasoningEffort represents the effort level for reasoning
type ReasoningEffort string

const (
	// ReasoningEffortXHigh allocates the largest portion of tokens for reasoning (~95% of max_tokens)
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
	// ReasoningEffortHigh allocates a large portion of tokens for reasoning (~80% of max_tokens)
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortMedium allocates a moderate portion of tokens (~50% of max_tokens)
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortLow allocates a smaller portion of tokens (~20% of max_tokens)
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMinimal allocates an even smaller portion of tokens (~10% of max_tokens)
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	// ReasoningEffortNone disables reasoning entirely
	ReasoningEffortNone ReasoningEffort = "none"
)

// ReasoningOptions configures reasoning token behavior for a chat completion request.
// The Effort and MaxTokens fields are mutually exclusive: use one or the other.
type ReasoningOptions struct {
	// Effort controls the reasoning effort level (OpenAI-style).
	// Supported by: OpenAI o-series, Grok models, Gemini 3 models.
	// For providers only supporting max_tokens, the effort level is mapped proportionally.
	Effort *ReasoningEffort

	// MaxTokens specifies the exact token budget for reasoning (Anthropic-style).
	// Supported by: Anthropic Claude reasoning models, Gemini thinking models.
	// For providers only supporting effort levels, this is mapped to the nearest level.
	MaxTokens *int

	// Exclude controls whether reasoning tokens are returned in the response.
	// When true, the model still uses reasoning internally but does not include it in the response.
	// Default is false (reasoning is included in the response when available).
	Exclude bool

	// Enabled enables reasoning with the default parameters (medium effort, no exclusions).
	// Default is inferred from Effort or MaxTokens.
	Enabled *bool
}

func NewReasoningOptions(effort ReasoningEffort) *ReasoningOptions {
	enabled := true
	return &ReasoningOptions{
		Effort:  &effort,
		Enabled: &enabled,
	}
}

// ReasoningDetailType represents the type of a reasoning detail block
type ReasoningDetailType string

const (
	// ReasoningDetailTypeText contains raw text reasoning with optional signature verification
	ReasoningDetailTypeText ReasoningDetailType = "reasoning.text"
	// ReasoningDetailTypeSummary contains a high-level summary of the reasoning process
	ReasoningDetailTypeSummary ReasoningDetailType = "reasoning.summary"
	// ReasoningDetailTypeEncrypted contains encrypted reasoning data (e.g. redacted blocks)
	ReasoningDetailTypeEncrypted ReasoningDetailType = "reasoning.encrypted"
)

// ReasoningDetail represents a single reasoning detail block from an LLM response.
// When preserving reasoning across turns, pass the entire ReasoningDetails slice back
// unmodified in the next request's assistant message.
type ReasoningDetail struct {
	// ID is the unique identifier for this reasoning detail block
	ID string
	// Type is the type of reasoning detail (text, summary, or encrypted)
	Type ReasoningDetailType
	// Text contains raw text reasoning (populated when Type is ReasoningDetailTypeText)
	Text string
	// Summary contains a high-level summary (populated when Type is ReasoningDetailTypeSummary)
	Summary string
	// Data contains encrypted reasoning data (populated when Type is ReasoningDetailTypeEncrypted)
	Data string
	// Format identifies the provider-specific format of this reasoning detail
	Format string
	// Index is the sequential index of this reasoning detail within the response
	Index int
	// Signature is used for signature verification of text reasoning blocks
	Signature string
}

// ReasoningMessage extends Message with reasoning capabilities.
// Implementations carry reasoning content that should be preserved across
// multi-turn conversations, especially when tool calls are involved.
type ReasoningMessage interface {
	Message
	// Reasoning returns the plaintext reasoning string.
	// This is the simpler form and can be used with models that return raw reasoning strings.
	Reasoning() string
	// ReasoningDetails returns the structured reasoning detail blocks.
	// Use this when working with models that return encrypted or summarized reasoning blocks,
	// as it preserves the full structure needed by those models.
	ReasoningDetails() []ReasoningDetail
}

// BaseAssistantReasoningMessage is an assistant message that carries reasoning content
// for multi-turn preservation.
type BaseAssistantReasoningMessage struct {
	BaseMessage
	reasoning        string
	reasoningDetails []ReasoningDetail
}

// Reasoning implements ReasoningMessage
func (m *BaseAssistantReasoningMessage) Reasoning() string {
	return m.reasoning
}

// ReasoningDetails implements ReasoningMessage
func (m *BaseAssistantReasoningMessage) ReasoningDetails() []ReasoningDetail {
	return m.reasoningDetails
}

// Attachments implements Message
func (m *BaseAssistantReasoningMessage) Attachments() []Attachment {
	return nil
}

var _ ReasoningMessage = &BaseAssistantReasoningMessage{}

// NewAssistantReasoningMessage creates a new assistant message that carries reasoning
// content for multi-turn preservation.
func NewAssistantReasoningMessage(content, reasoning string, details []ReasoningDetail) *BaseAssistantReasoningMessage {
	return &BaseAssistantReasoningMessage{
		BaseMessage: BaseMessage{
			role:    RoleAssistant,
			content: content,
		},
		reasoning:        reasoning,
		reasoningDetails: details,
	}
}

// ReasoningChatCompletionResponse extends ChatCompletionResponse with reasoning content.
// Providers that support reasoning implement this interface to expose the reasoning
// tokens returned by the model. Callers can type-assert to access reasoning data.
type ReasoningChatCompletionResponse interface {
	ChatCompletionResponse
	// Reasoning returns the plaintext reasoning string from the response.
	Reasoning() string
	// ReasoningDetails returns the structured reasoning detail blocks from the response.
	ReasoningDetails() []ReasoningDetail
}

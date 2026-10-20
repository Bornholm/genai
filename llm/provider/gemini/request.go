package gemini

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

// signatureFormat marks the reasoning details carrying thought signatures,
// with the value OpenRouter gives them.
const signatureFormat = "google-gemini-v1"

// skipSignature is the placeholder Google documents for a function call
// replayed without its thought signature, typically one produced by another
// provider: Gemini 3 refuses a function call turn without one.
const skipSignature = "skip_thought_signature_validator"

// request is the body of generateContent. Only the fields genai has an
// option for are typed; ExtraFields travel through MarshalJSON.
type request struct {
	Contents          []content         `json:"contents"`
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
	Tools             []toolSet         `json:"tools,omitempty"`
	ToolConfig        *toolConfig       `json:"toolConfig,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
	extra             map[string]any
}

// MarshalJSON merges the extra fields over the typed ones, which they
// override, the way the other providers merge ExtraFields.
func (r request) MarshalJSON() ([]byte, error) {
	type plain request
	raw, err := json.Marshal(plain(r))
	if err != nil || len(r.extra) == 0 {
		return raw, err
	}
	var merged map[string]any
	if err := json.Unmarshal(raw, &merged); err != nil {
		return nil, err
	}
	for k, v := range r.extra {
		merged[k] = v
	}
	return json.Marshal(merged)
}

type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type part struct {
	Text             string            `json:"text,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
	InlineData       *blob             `json:"inlineData,omitempty"`
	FileData         *fileData         `json:"fileData,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
}

type blob struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type fileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

type functionCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type functionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type toolSet struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations"`
}

type functionDeclaration struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	// ParametersJSONSchema takes a plain JSON schema, where parameters
	// would demand Google's OpenAPI subset.
	ParametersJSONSchema map[string]any `json:"parametersJsonSchema,omitempty"`
}

type toolConfig struct {
	FunctionCallingConfig functionCallingConfig `json:"functionCallingConfig"`
}

type functionCallingConfig struct {
	Mode string `json:"mode"`
}

type generationConfig struct {
	Temperature        *float64        `json:"temperature,omitempty"`
	MaxOutputTokens    *int            `json:"maxOutputTokens,omitempty"`
	Seed               *int            `json:"seed,omitempty"`
	ResponseMimeType   string          `json:"responseMimeType,omitempty"`
	ResponseJSONSchema any             `json:"responseJsonSchema,omitempty"`
	ThinkingConfig     *thinkingConfig `json:"thinkingConfig,omitempty"`
}

type thinkingConfig struct {
	ThinkingBudget  *int `json:"thinkingBudget,omitempty"`
	IncludeThoughts bool `json:"includeThoughts,omitempty"`
}

// dynamicBudget lets the model pick its thinking budget.
const dynamicBudget = -1

// effortBudgets is the thinking budget of each effort level when no
// max_output_tokens is set to take a share of.
var effortBudgets = map[llm.ReasoningEffort]int{
	llm.ReasoningEffortMinimal: minEffortBudget,
	llm.ReasoningEffortLow:     1024,
	llm.ReasoningEffortMedium:  8192,
	llm.ReasoningEffortHigh:    16384,
	llm.ReasoningEffortXHigh:   maxEffortBudget,
}

// minEffortBudget and maxEffortBudget bound a budget derived from an effort
// level to the range every thinking model accepts: Flash-Lite needs at least
// 512 tokens, Flash takes at most 24576.
const (
	minEffortBudget = 512
	maxEffortBudget = 24576
)

// effortRatios is the share of max_tokens each effort stands for, as
// documented on llm.ReasoningEffort.
var effortRatios = map[llm.ReasoningEffort]float64{
	llm.ReasoningEffortMinimal: 0.10,
	llm.ReasoningEffortLow:     0.20,
	llm.ReasoningEffortMedium:  0.50,
	llm.ReasoningEffortHigh:    0.80,
	llm.ReasoningEffortXHigh:   0.95,
}

// buildRequest converts genai chat completion options to a generateContent
// request.
func buildRequest(opts *llm.ChatCompletionOptions) (*request, error) {
	if opts.Audio != nil {
		return nil, llm.NewValidationError("audio", "the Gemini provider has no audio output")
	}
	for _, modality := range opts.Modalities {
		if modality != "text" {
			return nil, llm.NewValidationError("modalities", fmt.Sprintf("the Gemini provider has no %q output", modality))
		}
	}

	system, contents, err := buildContents(opts.Messages)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	if len(contents) == 0 {
		return nil, llm.NewValidationError("messages", "at least one non-system message is required")
	}

	req := &request{
		Contents:          contents,
		SystemInstruction: system,
		extra:             opts.ExtraFields,
	}

	config := &generationConfig{
		Temperature:     opts.Temperature,
		MaxOutputTokens: opts.MaxCompletionTokens,
		Seed:            opts.Seed,
		ThinkingConfig:  buildThinking(opts.Reasoning, opts.MaxCompletionTokens),
	}

	if len(opts.Tools) > 0 {
		declarations := make([]functionDeclaration, 0, len(opts.Tools))
		for _, t := range opts.Tools {
			declarations = append(declarations, functionDeclaration{
				Name:                 t.Name(),
				Description:          t.Description(),
				ParametersJSONSchema: t.Parameters(),
			})
		}
		req.Tools = []toolSet{{FunctionDeclarations: declarations}}

		switch opts.ToolChoice {
		case llm.ToolChoiceAuto:
			req.ToolConfig = &toolConfig{FunctionCallingConfig: functionCallingConfig{Mode: "AUTO"}}
		case llm.ToolChoiceRequired:
			req.ToolConfig = &toolConfig{FunctionCallingConfig: functionCallingConfig{Mode: "ANY"}}
		case llm.ToolChoiceNone:
			req.ToolConfig = &toolConfig{FunctionCallingConfig: functionCallingConfig{Mode: "NONE"}}
		}
	}

	if opts.ResponseFormat == llm.ResponseFormatJSON {
		config.ResponseMimeType = "application/json"
		if opts.ResponseSchema != nil {
			config.ResponseJSONSchema = opts.ResponseSchema.Schema()
		}
	}

	if config.Temperature != nil || config.MaxOutputTokens != nil || config.Seed != nil ||
		config.ResponseMimeType != "" || config.ThinkingConfig != nil {
		req.GenerationConfig = config
	}

	return req, nil
}

// buildThinking maps llm.ReasoningOptions to a thinking budget. An explicit
// budget is sent as is. An effort level becomes a share of
// max_output_tokens, or a fixed budget when none is set, bounded to what
// every thinking model accepts; reasoning enabled without either lets the
// model choose. Disabled reasoning is a zero budget, which the models that
// cannot stop thinking (2.5 Pro, Gemini 3) refuse. The budget is what
// Gemini 2.5 takes, and what Gemini 3 still accepts in place of its
// thinking levels.
func buildThinking(opts *llm.ReasoningOptions, maxOutputTokens *int) *thinkingConfig {
	if opts == nil {
		return nil
	}

	var budget int
	switch {
	case opts.Enabled != nil && !*opts.Enabled, opts.Effort != nil && *opts.Effort == llm.ReasoningEffortNone:
		budget = 0
	case opts.MaxTokens != nil:
		budget = *opts.MaxTokens
	case opts.Effort != nil:
		effort := *opts.Effort
		if _, known := effortBudgets[effort]; !known {
			effort = llm.ReasoningEffortMedium
		}
		if maxOutputTokens != nil && *maxOutputTokens > 0 {
			budget = int(float64(*maxOutputTokens) * effortRatios[effort])
			budget = min(max(budget, minEffortBudget), maxEffortBudget)
		} else {
			budget = effortBudgets[effort]
		}
	case opts.Enabled != nil && *opts.Enabled:
		budget = dynamicBudget
	case opts.Exclude:
		// The model keeps its default budget; its thoughts are still asked
		// for, for their signatures.
		return &thinkingConfig{IncludeThoughts: true}
	default:
		return nil
	}

	// Thoughts are asked for even when excluded, and only withheld from
	// the caller: the signatures of a turn must come back whatever the
	// caller wants to read.
	return &thinkingConfig{
		ThinkingBudget:  &budget,
		IncludeThoughts: budget != 0,
	}
}

// buildContents converts genai messages: system messages are hoisted into
// the system instruction, the rest becomes user and model turns. Tool
// results join the previous turn when it holds tool results, so that the
// answers to parallel function calls share one turn as the API requires; a
// user message after them opens a turn of its own. Model messages that
// follow each other share one turn.
//
// Cache hints are dropped: the API caches prompt prefixes on its own.
func buildContents(msgs []llm.Message) (*content, []content, error) {
	var (
		system   *content
		contents []content
		// names maps a function call ID to its name: a function response
		// must name its function, which a genai tool message does not.
		names = map[string]string{}
	)

	// toolTurn marks the index of the last turn when it holds tool results.
	toolTurn := -1
	add := func(role string, parts []part, toolResults bool) {
		if len(parts) == 0 {
			return
		}
		if n := len(contents); n > 0 && contents[n-1].Role == role && (toolTurn == n-1) == toolResults {
			contents[n-1].Parts = append(contents[n-1].Parts, parts...)
			return
		}
		contents = append(contents, content{Role: role, Parts: parts})
		if toolResults {
			toolTurn = len(contents) - 1
		}
	}

	for _, m := range msgs {
		switch m.Role() {
		case llm.RoleSystem:
			if len(m.Attachments()) > 0 {
				return nil, nil, errors.New("system messages cannot have attachments")
			}
			if m.Content() == "" {
				continue
			}
			if system == nil {
				system = &content{}
			}
			system.Parts = append(system.Parts, part{Text: m.Content()})

		case llm.RoleUser:
			parts := make([]part, 0, 1+len(m.Attachments()))
			if text := m.Content(); text != "" {
				parts = append(parts, part{Text: text})
			}
			for _, attachment := range m.Attachments() {
				p, err := attachmentPart(attachment)
				if err != nil {
					return nil, nil, errors.WithStack(err)
				}
				parts = append(parts, p)
			}
			add("user", parts, false)

		case llm.RoleAssistant:
			if len(m.Attachments()) > 0 {
				return nil, nil, errors.New("assistant messages cannot have attachments")
			}
			if m.Content() == "" {
				return nil, nil, llm.NewValidationError("messages", "assistant message has no content to send")
			}
			signatures := thoughtSignatures(m)
			add("model", []part{{Text: m.Content(), ThoughtSignature: signatures[""]}}, false)

		case llm.RoleToolCalls:
			if len(m.Attachments()) > 0 {
				return nil, nil, errors.New("tool calls messages cannot have attachments")
			}
			toolCallsMessage, ok := m.(llm.ToolCallsMessage)
			if !ok {
				return nil, nil, errors.Errorf("unexpected tool calls message type '%T'", m)
			}
			signatures := thoughtSignatures(m)
			var parts []part
			if text := m.Content(); text != "" {
				parts = append(parts, part{Text: text, ThoughtSignature: signatures[""]})
			}
			for i, tc := range toolCallsMessage.ToolCalls() {
				args, err := toolCallArgs(tc.Parameters())
				if err != nil {
					return nil, nil, errors.Wrapf(err, "invalid parameters for tool call '%s'", tc.ID())
				}
				names[tc.ID()] = tc.Name()
				p := part{
					FunctionCall:     &functionCall{ID: tc.ID(), Name: tc.Name(), Args: args},
					ThoughtSignature: signatures[tc.ID()],
				}
				// Google validates the signature of the first function call
				// of a step only: parallel calls after it carry none.
				if i == 0 && p.ThoughtSignature == "" {
					p.ThoughtSignature = skipSignature
				}
				parts = append(parts, p)
			}
			add("model", parts, false)

		case llm.RoleTool:
			toolMessage, ok := m.(llm.ToolMessage)
			if !ok {
				return nil, nil, errors.Errorf("unexpected tool message type '%T'", m)
			}
			name, known := names[toolMessage.ID()]
			if !known {
				return nil, nil, llm.NewValidationError("messages", fmt.Sprintf("tool result '%s' answers no earlier tool call", toolMessage.ID()))
			}
			add("user", toolResultParts(toolMessage, name), true)

		default:
			return nil, nil, errors.Errorf("unsupported message role '%s'", m.Role())
		}
	}

	return system, contents, nil
}

// toolResultParts builds the function response and, after it, the parts
// carrying the attachments the tool returned. An attachment the API cannot
// carry is replaced by a note rather than failing the request: the caller
// did not choose what the tool returned.
func toolResultParts(m llm.ToolMessage, name string) []part {
	parts := []part{{FunctionResponse: &functionResponse{
		ID:       m.ID(),
		Name:     name,
		Response: toolResponse(m.Content()),
	}}}
	for i, attachment := range m.Attachments() {
		p, err := attachmentPart(attachment)
		if err != nil {
			slog.Warn("gemini: leaving out a tool result attachment the provider cannot carry",
				slog.String("call_id", m.ID()), slog.Int("index", i),
				slog.String("type", string(attachment.Type())), slog.String("mime_type", attachment.MimeType()),
				slog.String("error", err.Error()))
			p = part{Text: llm.OmittedAttachmentNote(attachment)}
		}
		parts = append(parts, p)
	}
	return parts
}

// toolResponse wraps a tool result in the object a function response
// carries: a result that is itself a JSON object is sent as is, anything
// else under "result".
func toolResponse(result string) map[string]any {
	var object map[string]any
	if err := json.Unmarshal([]byte(result), &object); err == nil && object != nil {
		return object
	}
	return map[string]any{"result": result}
}

// thoughtSignatures returns the thought signatures this provider attached to
// a message, by function call ID; the signature of a text part has an empty
// ID. Reasoning from another provider has no signature to give.
func thoughtSignatures(m llm.Message) map[string]string {
	signatures := map[string]string{}
	rm, ok := m.(llm.ReasoningMessage)
	if !ok {
		return signatures
	}
	for _, d := range rm.ReasoningDetails() {
		if d.Format == signatureFormat && d.Type == llm.ReasoningDetailTypeEncrypted && d.Data != "" {
			signatures[d.ID] = d.Data
		}
	}
	return signatures
}

// toolCallArgs decodes the tool call parameters, kept as a JSON string by
// genai, into the object a function call carries. Arguments that are not a
// JSON object, as some models emit, are replayed as an empty object, as the
// openai provider does: rejecting them would block the conversation.
func toolCallArgs(parameters any) (map[string]any, error) {
	var raw []byte
	switch typed := parameters.(type) {
	case nil:
		return map[string]any{}, nil
	case string:
		raw = []byte(typed)
	case []byte:
		raw = typed
	case map[string]any:
		return typed, nil
	default:
		encoded, err := json.Marshal(typed)
		if err != nil {
			return nil, errors.WithStack(err)
		}
		raw = encoded
	}
	if len(raw) == 0 {
		return map[string]any{}, nil
	}
	var args map[string]any
	if err := json.Unmarshal(raw, &args); err != nil || args == nil {
		return map[string]any{}, nil
	}
	return args, nil
}

// attachmentPart converts an attachment to a part: inline data when it is
// base64, file data when it is a URL. Images, audio, video, PDF and text
// documents are all understood by Gemini models.
func attachmentPart(attachment llm.Attachment) (part, error) {
	mimeType := normalizeMimeType(attachment.MimeType())

	switch attachment.Type() {
	case llm.AttachmentTypeImage, llm.AttachmentTypeAudio, llm.AttachmentTypeVideo:
	case llm.AttachmentTypeDocument:
		if mimeType != "application/pdf" && !strings.HasPrefix(mimeType, "text/") {
			return part{}, errors.Errorf("unsupported document MIME type: %s (supported: application/pdf, text/*)", attachment.MimeType())
		}
	default:
		return part{}, errors.Errorf("unsupported attachment type '%s'", attachment.Type())
	}

	switch attachment.Source() {
	case llm.AttachmentSourceBase64:
		data, err := stripDataURL(attachment.Data())
		if err != nil {
			return part{}, err
		}
		if _, err := base64.StdEncoding.DecodeString(data); err != nil {
			return part{}, errors.Wrap(err, "invalid base64 attachment")
		}
		return part{InlineData: &blob{MimeType: mimeType, Data: data}}, nil
	case llm.AttachmentSourceURL:
		return part{FileData: &fileData{MimeType: mimeType, FileURI: attachment.Data()}}, nil
	}

	return part{}, errors.Errorf("unsupported attachment source '%s'", attachment.Source())
}

// normalizeMimeType lowercases a MIME type, drops its parameters and maps
// the common image/jpg misspelling to its registered name.
func normalizeMimeType(mimeType string) string {
	base, _, _ := strings.Cut(mimeType, ";")
	base = strings.ToLower(strings.TrimSpace(base))
	if base == "image/jpg" {
		return "image/jpeg"
	}
	return base
}

// stripDataURL returns the payload of a "data:<mime>;base64,<payload>" URL,
// or the input unchanged when it is already a bare base64 string.
func stripDataURL(data string) (string, error) {
	if !strings.HasPrefix(data, "data:") {
		return data, nil
	}
	header, payload, found := strings.Cut(data, ",")
	if !found {
		return "", errors.New("malformed data URL: no payload")
	}
	if !strings.HasSuffix(header, ";base64") {
		return "", errors.New("unsupported data URL: only base64-encoded payloads are accepted")
	}
	return payload, nil
}

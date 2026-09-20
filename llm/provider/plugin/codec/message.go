// Package codec converts between the llm types and their protobuf wire form.
// It is shared by the host side (llm/provider/plugin) and the plugin side
// (plugin/sdk), so that both agree on which llm constructor a given set of
// fields maps to.
package codec

import (
	"github.com/bornholm/genai/llm"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/pkg/errors"
)

var roleToProto = map[llm.Role]pluginv1.Role{
	llm.RoleUser:      pluginv1.Role_ROLE_USER,
	llm.RoleSystem:    pluginv1.Role_ROLE_SYSTEM,
	llm.RoleAssistant: pluginv1.Role_ROLE_ASSISTANT,
	llm.RoleTool:      pluginv1.Role_ROLE_TOOL,
	llm.RoleToolCalls: pluginv1.Role_ROLE_TOOL_CALLS,
}

var roleFromProto = map[pluginv1.Role]llm.Role{
	pluginv1.Role_ROLE_USER:       llm.RoleUser,
	pluginv1.Role_ROLE_SYSTEM:     llm.RoleSystem,
	pluginv1.Role_ROLE_ASSISTANT:  llm.RoleAssistant,
	pluginv1.Role_ROLE_TOOL:       llm.RoleTool,
	pluginv1.Role_ROLE_TOOL_CALLS: llm.RoleToolCalls,
}

// RoleToProto converts a role; an unknown role maps to ROLE_UNSPECIFIED.
func RoleToProto(role llm.Role) pluginv1.Role {
	return roleToProto[role]
}

// RoleFromProto converts a role. An unspecified or unknown role is an
// error: a plugin speaking another version of the protocol should fail at
// the edge, not produce a message in-process providers reject later.
func RoleFromProto(role pluginv1.Role) (llm.Role, error) {
	result, ok := roleFromProto[role]
	if !ok {
		return "", errors.Errorf("unknown role %q", role.String())
	}
	return result, nil
}

var attachmentTypeToProto = map[llm.AttachmentType]pluginv1.AttachmentType{
	llm.AttachmentTypeImage:    pluginv1.AttachmentType_ATTACHMENT_TYPE_IMAGE,
	llm.AttachmentTypeAudio:    pluginv1.AttachmentType_ATTACHMENT_TYPE_AUDIO,
	llm.AttachmentTypeVideo:    pluginv1.AttachmentType_ATTACHMENT_TYPE_VIDEO,
	llm.AttachmentTypeDocument: pluginv1.AttachmentType_ATTACHMENT_TYPE_DOCUMENT,
}

var attachmentTypeFromProto = map[pluginv1.AttachmentType]llm.AttachmentType{
	pluginv1.AttachmentType_ATTACHMENT_TYPE_IMAGE:    llm.AttachmentTypeImage,
	pluginv1.AttachmentType_ATTACHMENT_TYPE_AUDIO:    llm.AttachmentTypeAudio,
	pluginv1.AttachmentType_ATTACHMENT_TYPE_VIDEO:    llm.AttachmentTypeVideo,
	pluginv1.AttachmentType_ATTACHMENT_TYPE_DOCUMENT: llm.AttachmentTypeDocument,
}

// AttachmentToProto converts an attachment.
func AttachmentToProto(attachment llm.Attachment) *pluginv1.Attachment {
	source := pluginv1.AttachmentSource_ATTACHMENT_SOURCE_BASE64
	if attachment.Source() == llm.AttachmentSourceURL {
		source = pluginv1.AttachmentSource_ATTACHMENT_SOURCE_URL
	}
	return &pluginv1.Attachment{
		Type:     attachmentTypeToProto[attachment.Type()],
		MimeType: attachment.MimeType(),
		Source:   source,
		Data:     attachment.Data(),
	}
}

// AttachmentFromProto rebuilds an attachment, validating it the way the llm
// constructors do.
func AttachmentFromProto(attachment *pluginv1.Attachment) (llm.Attachment, error) {
	attachmentType, ok := attachmentTypeFromProto[attachment.GetType()]
	if !ok {
		return nil, errors.Errorf("unknown attachment type %q", attachment.GetType().String())
	}
	switch attachment.GetSource() {
	case pluginv1.AttachmentSource_ATTACHMENT_SOURCE_URL:
		result, err := llm.NewURLAttachment(attachmentType, attachment.GetMimeType(), attachment.GetData())
		if err != nil {
			return nil, errors.WithStack(err)
		}
		return result, nil
	case pluginv1.AttachmentSource_ATTACHMENT_SOURCE_BASE64:
		result, err := llm.NewBase64Attachment(attachmentType, attachment.GetMimeType(), attachment.GetData())
		if err != nil {
			return nil, errors.WithStack(err)
		}
		return result, nil
	default:
		return nil, errors.Errorf("unknown attachment source %q", attachment.GetSource().String())
	}
}

func attachmentsToProto(attachments []llm.Attachment) []*pluginv1.Attachment {
	if len(attachments) == 0 {
		return nil
	}
	result := make([]*pluginv1.Attachment, 0, len(attachments))
	for _, a := range attachments {
		result = append(result, AttachmentToProto(a))
	}
	return result
}

func attachmentsFromProto(attachments []*pluginv1.Attachment) ([]llm.Attachment, error) {
	if len(attachments) == 0 {
		return nil, nil
	}
	result := make([]llm.Attachment, 0, len(attachments))
	for _, a := range attachments {
		attachment, err := AttachmentFromProto(a)
		if err != nil {
			return nil, errors.WithStack(err)
		}
		result = append(result, attachment)
	}
	return result, nil
}

// ReasoningDetailToProto converts a reasoning detail.
func ReasoningDetailToProto(detail llm.ReasoningDetail) *pluginv1.ReasoningDetail {
	return &pluginv1.ReasoningDetail{
		Id:        detail.ID,
		Type:      string(detail.Type),
		Text:      detail.Text,
		Summary:   detail.Summary,
		Data:      detail.Data,
		Format:    detail.Format,
		Index:     int32(detail.Index),
		Signature: detail.Signature,
	}
}

// ReasoningDetailFromProto converts a reasoning detail.
func ReasoningDetailFromProto(detail *pluginv1.ReasoningDetail) llm.ReasoningDetail {
	return llm.ReasoningDetail{
		ID:        detail.GetId(),
		Type:      llm.ReasoningDetailType(detail.GetType()),
		Text:      detail.GetText(),
		Summary:   detail.GetSummary(),
		Data:      detail.GetData(),
		Format:    detail.GetFormat(),
		Index:     int(detail.GetIndex()),
		Signature: detail.GetSignature(),
	}
}

func reasoningDetailsToProto(details []llm.ReasoningDetail) []*pluginv1.ReasoningDetail {
	if len(details) == 0 {
		return nil
	}
	result := make([]*pluginv1.ReasoningDetail, 0, len(details))
	for _, d := range details {
		result = append(result, ReasoningDetailToProto(d))
	}
	return result
}

func reasoningDetailsFromProto(details []*pluginv1.ReasoningDetail) []llm.ReasoningDetail {
	if len(details) == 0 {
		return nil
	}
	result := make([]llm.ReasoningDetail, 0, len(details))
	for _, d := range details {
		result = append(result, ReasoningDetailFromProto(d))
	}
	return result
}

// ToolCallToProto converts a tool call. Parameters are carried as JSON text,
// whatever their in-memory form.
func ToolCallToProto(toolCall llm.ToolCall) (*pluginv1.ToolCall, error) {
	params, err := parametersJSON(toolCall.Parameters())
	if err != nil {
		return nil, errors.Wrapf(err, "could not encode parameters of tool call %q", toolCall.ID())
	}
	return &pluginv1.ToolCall{
		Id:             toolCall.ID(),
		Name:           toolCall.Name(),
		ParametersJson: params,
	}, nil
}

// ToolCallFromProto converts a tool call.
func ToolCallFromProto(toolCall *pluginv1.ToolCall) llm.ToolCall {
	return llm.NewToolCall(toolCall.GetId(), toolCall.GetName(), toolCall.GetParametersJson())
}

func toolCallsToProto(toolCalls []llm.ToolCall) ([]*pluginv1.ToolCall, error) {
	if len(toolCalls) == 0 {
		return nil, nil
	}
	result := make([]*pluginv1.ToolCall, 0, len(toolCalls))
	for _, tc := range toolCalls {
		converted, err := ToolCallToProto(tc)
		if err != nil {
			return nil, errors.WithStack(err)
		}
		result = append(result, converted)
	}
	return result, nil
}

func toolCallsFromProto(toolCalls []*pluginv1.ToolCall) []llm.ToolCall {
	if len(toolCalls) == 0 {
		return nil
	}
	result := make([]llm.ToolCall, 0, len(toolCalls))
	for _, tc := range toolCalls {
		result = append(result, ToolCallFromProto(tc))
	}
	return result
}

// MessageToProto converts a message, whatever its concrete type, by probing
// the optional interfaces it implements.
func MessageToProto(message llm.Message) (*pluginv1.Message, error) {
	if message == nil {
		return nil, errors.New("message is nil")
	}
	result := &pluginv1.Message{
		Role:        RoleToProto(message.Role()),
		Content:     message.Content(),
		Attachments: attachmentsToProto(message.Attachments()),
	}

	if cc, ok := message.(llm.CacheControlMessage); ok {
		if control := cc.CacheControl(); control != nil {
			result.CacheControl = &pluginv1.CacheControl{Type: control.Type}
			if control.TTL != nil {
				ttl := *control.TTL
				result.CacheControl.Ttl = &ttl
			}
		}
	}

	if tm, ok := message.(llm.ToolMessage); ok {
		result.ToolCallId = tm.ID()
	}

	if tcm, ok := message.(llm.ToolCallsMessage); ok {
		toolCalls, err := toolCallsToProto(tcm.ToolCalls())
		if err != nil {
			return nil, errors.WithStack(err)
		}
		result.ToolCalls = toolCalls
	}

	if rm, ok := message.(llm.ReasoningMessage); ok {
		result.Reasoning = rm.Reasoning()
		result.ReasoningDetails = reasoningDetailsToProto(rm.ReasoningDetails())
	}

	return result, nil
}

// MessageFromProto rebuilds a message, choosing the llm constructor from the
// fields that are set. Tool call and reasoning messages cannot carry
// attachments in llm (their Attachments() is nil), so the wire form never
// has any for them; every other combination is preserved.
func MessageFromProto(message *pluginv1.Message) (llm.Message, error) {
	role, err := RoleFromProto(message.GetRole())
	if err != nil {
		return nil, errors.WithStack(err)
	}
	attachments, err := attachmentsFromProto(message.GetAttachments())
	if err != nil {
		return nil, errors.WithStack(err)
	}

	var cacheControl *llm.CacheControl
	if cc := message.GetCacheControl(); cc != nil {
		cacheControl = &llm.CacheControl{Type: cc.GetType()}
		if cc.Ttl != nil {
			ttl := cc.GetTtl()
			cacheControl.TTL = &ttl
		}
	}

	var result llm.Message
	switch {
	case role == llm.RoleTool || message.GetToolCallId() != "":
		result = llm.NewToolMessage(message.GetToolCallId(), llm.NewToolResult(message.GetContent(), attachments...))

	case role == llm.RoleToolCalls || len(message.GetToolCalls()) > 0:
		if len(attachments) > 0 {
			return nil, errors.New("a tool calls message cannot carry attachments")
		}
		if message.GetReasoning() != "" || len(message.GetReasoningDetails()) > 0 {
			result = llm.NewReasoningToolCallsMessageWithContent(
				message.GetContent(),
				message.GetReasoning(),
				reasoningDetailsFromProto(message.GetReasoningDetails()),
				toolCallsFromProto(message.GetToolCalls())...,
			)
		} else {
			result = llm.NewToolCallsMessageWithContent(message.GetContent(), toolCallsFromProto(message.GetToolCalls())...)
		}

	case message.GetReasoning() != "" || len(message.GetReasoningDetails()) > 0:
		if len(attachments) > 0 {
			return nil, errors.New("a reasoning message cannot carry attachments")
		}
		result = llm.NewAssistantReasoningMessage(
			message.GetContent(),
			message.GetReasoning(),
			reasoningDetailsFromProto(message.GetReasoningDetails()),
		)

	case len(attachments) > 0:
		result = llm.NewMultimodalMessage(role, message.GetContent(), attachments...)

	default:
		result = llm.NewMessage(role, message.GetContent())
	}

	if cacheControl != nil && !llm.SetCacheControl(result, cacheControl) {
		return nil, errors.Errorf("message of type %T cannot carry a cache control", result)
	}
	return result, nil
}

func messagesToProto(messages []llm.Message) ([]*pluginv1.Message, error) {
	result := make([]*pluginv1.Message, 0, len(messages))
	for _, m := range messages {
		converted, err := MessageToProto(m)
		if err != nil {
			return nil, errors.WithStack(err)
		}
		result = append(result, converted)
	}
	return result, nil
}

func messagesFromProto(messages []*pluginv1.Message) ([]llm.Message, error) {
	result := make([]llm.Message, 0, len(messages))
	for _, m := range messages {
		converted, err := MessageFromProto(m)
		if err != nil {
			return nil, errors.WithStack(err)
		}
		result = append(result, converted)
	}
	return result, nil
}

package openrouter

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/bornholm/genai/llm"
	"github.com/revrost/go-openrouter"
)

const pngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

// wireMessages builds the messages of a request and decodes them back from
// their JSON form, so the assertions see what is sent.
func wireMessages(t *testing.T, msgs ...llm.Message) []map[string]any {
	t.Helper()

	messages, err := buildMessages(msgs, "anthropic/claude-sonnet-4.5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	raw, err := json.Marshal(messages)
	if err != nil {
		t.Fatalf("could not marshal messages: %v", err)
	}

	var decoded []map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("could not decode messages: %v", err)
	}

	return decoded
}

// cacheControls returns the cache_control of each content part of message,
// nil for a part without one, or nil when the content is not a part list.
func cacheControls(message map[string]any) []any {
	parts, ok := message["content"].([]any)
	if !ok {
		return nil
	}
	controls := make([]any, len(parts))
	for i, part := range parts {
		controls[i] = part.(map[string]any)["cache_control"]
	}
	return controls
}

func assertOnlyLastPartCached(t *testing.T, message map[string]any) {
	t.Helper()

	controls := cacheControls(message)
	if len(controls) == 0 {
		t.Fatalf("%s message content has no parts: %#v", message["role"], message["content"])
	}
	for i, cc := range controls[:len(controls)-1] {
		if cc != nil {
			t.Errorf("part %d has cache_control %v, want none", i, cc)
		}
	}
	last, _ := controls[len(controls)-1].(map[string]any)
	if last["type"] != "ephemeral" {
		t.Errorf("last part cache_control = %v, want ephemeral", controls[len(controls)-1])
	}
}

func ephemeral() *llm.CacheControl {
	return &llm.CacheControl{Type: "ephemeral"}
}

func TestBuildMessages_ToolResultCarriesCacheControl(t *testing.T) {
	toolMessage := llm.NewToolMessage("call_01", llm.NewToolResult("done"))
	llm.SetCacheControl(toolMessage, ephemeral())

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Run it"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "run", "{}")),
		toolMessage,
	)

	last := messages[len(messages)-1]
	if last["role"] != "tool" || last["tool_call_id"] != "call_01" {
		t.Fatalf("last message = %#v, want the tool result", last)
	}
	assertOnlyLastPartCached(t, last)
}

func TestBuildMessages_ToolResultWithAttachmentsCarriesCacheControl(t *testing.T) {
	image := pngAttachment(t)

	for name, attachments := range map[string][]llm.Attachment{
		"one attachment":  {image},
		"two attachments": {image, image},
	} {
		t.Run(name, func(t *testing.T) {
			toolMessage := llm.NewToolMessage("call_01", llm.NewToolResult("screenshot", attachments...))
			llm.SetCacheControl(toolMessage, ephemeral())

			messages := wireMessages(t,
				llm.NewMessage(llm.RoleUser, "Look"),
				llm.NewToolCallsMessage(llm.NewToolCall("call_01", "screenshot", "{}")),
				toolMessage,
			)

			assertOnlyLastPartCached(t, messages[len(messages)-1])
		})
	}
}

func TestBuildMessages_UserAttachmentsCarryCacheControl(t *testing.T) {
	image := pngAttachment(t)

	for name, attachments := range map[string][]llm.Attachment{
		"one attachment":  {image},
		"two attachments": {image, image},
	} {
		t.Run(name, func(t *testing.T) {
			user := llm.NewMultimodalMessage(llm.RoleUser, "What is this?", attachments...)
			llm.SetCacheControl(user, ephemeral())

			messages := wireMessages(t, user)

			assertOnlyLastPartCached(t, messages[0])
		})
	}
}

func TestBuildMessages_ToolCallsSendTextAndCarryCacheControl(t *testing.T) {
	toolCalls := llm.NewToolCallsMessageWithContent("Running it.", llm.NewToolCall("call_01", "run", "{}"))
	llm.SetCacheControl(toolCalls, ephemeral())

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Run it"),
		toolCalls,
	)

	last := messages[len(messages)-1]
	if last["role"] != "assistant" || last["tool_calls"] == nil {
		t.Fatalf("last message = %#v, want the tool calls", last)
	}
	parts, _ := last["content"].([]any)
	if len(parts) != 1 || parts[0].(map[string]any)["text"] != "Running it." {
		t.Errorf("tool calls content = %#v, want the assistant text", last["content"])
	}
	assertOnlyLastPartCached(t, last)
}

func TestBuildMessages_ToolCallsTextWithoutCacheControl(t *testing.T) {
	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Run it"),
		llm.NewToolCallsMessageWithContent("Running it.", llm.NewToolCall("call_01", "run", "{}")),
	)

	if got := messages[len(messages)-1]["content"]; got != "Running it." {
		t.Errorf("tool calls content = %#v, want the plain assistant text", got)
	}
}

func TestBuildMessages_ToolCallsWithoutTextMoveCacheControlToPreviousMessage(t *testing.T) {
	toolCalls := llm.NewToolCallsMessage(llm.NewToolCall("call_01", "run", "{}"))
	llm.SetCacheControl(toolCalls, ephemeral())

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Run it"),
		toolCalls,
	)

	if _, ok := messages[1]["content"]; ok {
		t.Errorf("tool calls without text sent content %#v", messages[1]["content"])
	}
	assertOnlyLastPartCached(t, messages[0])
}

func TestBuildMessages_CacheControlWithNothingBeforeIsDropped(t *testing.T) {
	toolCalls := llm.NewToolCallsMessage(llm.NewToolCall("call_01", "run", "{}"))
	llm.SetCacheControl(toolCalls, ephemeral())

	messages := wireMessages(t, toolCalls)

	if _, ok := messages[0]["content"]; ok {
		t.Errorf("tool calls without text sent content %#v", messages[0]["content"])
	}
}

func TestBuildMessages_UnmarkedMessagesStayPlainText(t *testing.T) {
	messages := wireMessages(t,
		llm.NewMessage(llm.RoleSystem, "Be brief"),
		llm.NewMessage(llm.RoleUser, "Run it"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "run", "{}")),
		llm.NewToolMessage("call_01", llm.NewToolResult("done")),
	)

	for _, m := range messages {
		if content, ok := m["content"]; ok {
			if _, isText := content.(string); !isText {
				t.Errorf("%s message content = %#v, want plain text", m["role"], content)
			}
		}
	}
}

func pngAttachment(t *testing.T) llm.Attachment {
	t.Helper()

	attachment, err := llm.NewBase64Attachment(llm.AttachmentTypeImage, "image/png", pngBase64)
	if err != nil {
		t.Fatalf("could not build attachment: %v", err)
	}
	return attachment
}

func textAttachment(t *testing.T, text string) llm.Attachment {
	t.Helper()

	attachment, err := llm.NewBase64Attachment(llm.AttachmentTypeDocument, "text/plain", base64.StdEncoding.EncodeToString([]byte(text)))
	if err != nil {
		t.Fatalf("could not build attachment: %v", err)
	}
	return attachment
}

func TestBuildMessages_SystemAndAssistantCarryCacheControl(t *testing.T) {
	messages := wireMessages(t,
		llm.NewMessageWithCacheControl(llm.RoleSystem, "Be brief", ephemeral()),
		llm.NewMessage(llm.RoleUser, "Hi"),
		llm.NewMessageWithCacheControl(llm.RoleAssistant, "Hello", ephemeral()),
	)

	assertOnlyLastPartCached(t, messages[0])
	assertOnlyLastPartCached(t, messages[2])
}

func TestBuildMessages_EmptyMessageMovesCacheControlToPreviousMessage(t *testing.T) {
	messages := wireMessages(t,
		llm.NewMessage(llm.RoleSystem, "Be brief"),
		llm.NewMessageWithCacheControl(llm.RoleUser, "", ephemeral()),
	)

	assertOnlyLastPartCached(t, messages[0])
	if _, ok := messages[1]["content"]; ok {
		t.Errorf("empty user message sent content %#v", messages[1]["content"])
	}
}

func TestBuildMessages_DocumentAttachmentCarriesCacheControl(t *testing.T) {
	user := llm.NewMultimodalMessage(llm.RoleUser, "Read this", textAttachment(t, "Some notes."))
	llm.SetCacheControl(user, ephemeral())

	messages := wireMessages(t, user)

	assertOnlyLastPartCached(t, messages[0])
}

func TestWithCacheControl_PassesOverEmptyTrailingTextParts(t *testing.T) {
	content, placed := withCacheControl(openrouter.Content{Multi: []openrouter.ChatMessagePart{
		{Type: openrouter.ChatMessagePartTypeText, Text: "Read this"},
		{Type: openrouter.ChatMessagePartTypeText, Text: ""},
	}}, ephemeral())

	if !placed {
		t.Fatal("hint not placed, want it on the text part")
	}
	if cc := content.Multi[0].CacheControl; cc == nil || cc.Type != "ephemeral" {
		t.Errorf("text part cache_control = %+v, want ephemeral", cc)
	}
	if cc := content.Multi[1].CacheControl; cc != nil {
		t.Errorf("empty part cache_control = %+v, want none", cc)
	}

	if _, placed := withCacheControl(openrouter.Content{Multi: []openrouter.ChatMessagePart{
		{Type: openrouter.ChatMessagePartTypeText, Text: ""},
	}}, ephemeral()); placed {
		t.Error("hint placed on an empty text part, want it reported as not placed")
	}
}

func TestBuildMessages_MovedCacheControlReplacesTheEarlierOne(t *testing.T) {
	hour, fiveMinutes := "1h", "5m"

	toolMessage := llm.NewToolMessage("call_01", llm.NewToolResult("done"))
	llm.SetCacheControl(toolMessage, &llm.CacheControl{Type: "ephemeral", TTL: &hour})

	toolCalls := llm.NewToolCallsMessage(llm.NewToolCall("call_02", "run", "{}"))
	llm.SetCacheControl(toolCalls, &llm.CacheControl{Type: "ephemeral", TTL: &fiveMinutes})

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Run it twice"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "run", "{}")),
		toolMessage,
		toolCalls,
	)

	controls := cacheControls(messages[2])
	if len(controls) != 1 {
		t.Fatalf("tool result content = %#v, want one part", messages[2]["content"])
	}
	if cc, _ := controls[0].(map[string]any); cc["ttl"] != "5m" {
		t.Errorf("tool result cache_control = %v, want the later hint with ttl 5m", controls[0])
	}
}

func TestBuildMessages_ToolCallsWithReasoningCarryCacheControl(t *testing.T) {
	toolCalls := llm.NewReasoningToolCallsMessageWithContent("Running it.", "The user wants it run.", nil, llm.NewToolCall("call_01", "run", "{}"))
	llm.SetCacheControl(toolCalls, ephemeral())

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Run it"),
		toolCalls,
	)

	last := messages[len(messages)-1]
	if last["reasoning"] != "The user wants it run." {
		t.Errorf("tool calls reasoning = %#v, want it preserved", last["reasoning"])
	}
	assertOnlyLastPartCached(t, last)
}

func TestBuildMessages_FallbackSkipsMessagesWithoutParts(t *testing.T) {
	toolCalls := llm.NewToolCallsMessage(llm.NewToolCall("call_02", "run", "{}"))
	llm.SetCacheControl(toolCalls, ephemeral())

	messages := wireMessages(t,
		llm.NewMessage(llm.RoleUser, "Run it twice"),
		llm.NewToolCallsMessage(llm.NewToolCall("call_01", "run", "{}")),
		toolCalls,
	)

	assertOnlyLastPartCached(t, messages[0])
	for _, m := range messages[1:] {
		if _, ok := m["content"]; ok {
			t.Errorf("tool calls without text sent content %#v", m["content"])
		}
	}
}

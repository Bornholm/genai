package codec

import (
	"context"
	"errors"
	"testing"

	"github.com/bornholm/genai/llm"
	pkgerrors "github.com/pkg/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestMessageRoundTrip(t *testing.T) {
	image, err := llm.NewImageAttachment("image/png", "https://example.com/a.png", true)
	if err != nil {
		t.Fatal(err)
	}
	ttl := "5m"
	details := []llm.ReasoningDetail{{ID: "r1", Type: llm.ReasoningDetailTypeText, Text: "because", Index: 1, Signature: "sig"}}

	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "sys"),
		llm.NewMessageWithCacheControl(llm.RoleUser, "cached", &llm.CacheControl{Type: "ephemeral", TTL: &ttl}),
		llm.NewMultimodalMessage(llm.RoleUser, "look", image),
		llm.NewMultimodalMessageWithCacheControl(llm.RoleUser, "cached look", &llm.CacheControl{Type: "ephemeral", TTL: &ttl}, image),
		llm.NewAssistantReasoningMessage("answer", "thought", details),
		llm.NewReasoningToolCallsMessageWithContent("calling", "why", details, llm.NewToolCall("c1", "tool", `{"a":1}`)),
		llm.NewToolMessage("c1", llm.NewToolResult("result", image)),
		withCacheControl(llm.NewToolMessage("c2", llm.NewToolResult("cached result")), ttl),
		withCacheControl(llm.NewToolCallsMessage(llm.NewToolCall("c3", "tool", `{"a":1}`)), ttl),
		withCacheControl(llm.NewAssistantReasoningMessage("cached answer", "thought", nil), ttl),
	}

	for i, original := range messages {
		encoded, err := MessageToProto(original)
		if err != nil {
			t.Fatalf("message %d: %+v", i, err)
		}
		decoded, err := MessageFromProto(encoded)
		if err != nil {
			t.Fatalf("message %d: %+v", i, err)
		}
		if decoded.Role() != original.Role() || decoded.Content() != original.Content() {
			t.Errorf("message %d: got %q/%q, want %q/%q", i, decoded.Role(), decoded.Content(), original.Role(), original.Content())
		}
		if len(decoded.Attachments()) != len(original.Attachments()) {
			t.Errorf("message %d: attachments %d != %d", i, len(decoded.Attachments()), len(original.Attachments()))
		}
		if cc, ok := original.(llm.CacheControlMessage); ok && cc.CacheControl() != nil {
			dcc, ok := decoded.(llm.CacheControlMessage)
			if !ok || dcc.CacheControl() == nil || *dcc.CacheControl().TTL != ttl {
				t.Errorf("message %d: cache control lost", i)
			}
		}
		if tm, ok := original.(llm.ToolMessage); ok {
			dtm, ok := decoded.(llm.ToolMessage)
			if !ok || dtm.ID() != tm.ID() {
				t.Errorf("message %d: tool message id lost", i)
			}
		}
		if tcm, ok := original.(llm.ToolCallsMessage); ok {
			dtcm, ok := decoded.(llm.ToolCallsMessage)
			if !ok || len(dtcm.ToolCalls()) != len(tcm.ToolCalls()) {
				t.Errorf("message %d: tool calls lost", i)
			} else if len(tcm.ToolCalls()) > 0 && dtcm.ToolCalls()[0].Parameters() != `{"a":1}` {
				t.Errorf("message %d: tool call parameters %v", i, dtcm.ToolCalls()[0].Parameters())
			}
		}
		if rm, ok := original.(llm.ReasoningMessage); ok {
			drm, ok := decoded.(llm.ReasoningMessage)
			if !ok || drm.Reasoning() != rm.Reasoning() || len(drm.ReasoningDetails()) != len(rm.ReasoningDetails()) {
				t.Errorf("message %d: reasoning lost", i)
			} else if len(rm.ReasoningDetails()) > 0 && drm.ReasoningDetails()[0] != rm.ReasoningDetails()[0] {
				t.Errorf("message %d: reasoning detail %+v != %+v", i, drm.ReasoningDetails()[0], rm.ReasoningDetails()[0])
			}
		}
	}
}

func withCacheControl(message llm.Message, ttl string) llm.Message {
	llm.SetCacheControl(message, &llm.CacheControl{Type: "ephemeral", TTL: &ttl})
	return message
}

func TestOptionsRoundTrip(t *testing.T) {
	schema := llm.NewResponseSchema("answer", "an answer", llm.NewJSONSchema().RequiredProperty("text", "text", "string")).WithStrict(false)
	tool := llm.NewFuncTool("get_weather", "weather", llm.NewJSONSchema().RequiredProperty("city", "city", "string"), nil)
	effort := llm.ReasoningEffortHigh

	original := llm.NewChatCompletionOptions(
		llm.WithMessages(llm.NewMessage(llm.RoleUser, "hi")),
		llm.WithTools(tool),
		llm.WithToolChoice(llm.ToolChoiceRequired),
		llm.WithTemperature(0.3),
		llm.WithSeed(42),
		llm.WithMaxCompletionTokens(100),
		llm.WithJSONResponse(schema),
		llm.WithReasoning(&llm.ReasoningOptions{Effort: &effort, Exclude: true}),
		llm.WithModalities("text", "audio"),
		llm.WithAudioOutput("alloy", "wav"),
		llm.WithSessionID("s1"),
		llm.WithExtraFields(map[string]any{"reasoning_split": true, "n": 2}),
	)

	encoded, err := ChatCompletionOptionsToProto(original)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	funcs, err := ChatCompletionOptionsFromProto(encoded)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	decoded := llm.NewChatCompletionOptions(funcs...)

	if len(decoded.Messages) != 1 || decoded.Messages[0].Content() != "hi" {
		t.Errorf("messages lost: %#v", decoded.Messages)
	}
	if len(decoded.Tools) != 1 || decoded.Tools[0].Name() != "get_weather" || decoded.Tools[0].Parameters()["type"] != "object" {
		t.Errorf("tools lost: %#v", decoded.Tools)
	}
	if _, err := decoded.Tools[0].Execute(t.Context(), nil); !errors.Is(err, ErrToolNotExecutable) {
		t.Errorf("expected ErrToolNotExecutable, got %v", err)
	}
	if decoded.ToolChoice != llm.ToolChoiceRequired {
		t.Errorf("tool choice %q", decoded.ToolChoice)
	}
	if decoded.Temperature == nil || *decoded.Temperature != 0.3 {
		t.Errorf("temperature %v", decoded.Temperature)
	}
	if decoded.Seed == nil || *decoded.Seed != 42 || decoded.MaxCompletionTokens == nil || *decoded.MaxCompletionTokens != 100 {
		t.Errorf("seed/max tokens lost")
	}
	if decoded.ResponseFormat != llm.ResponseFormatJSON || decoded.ResponseSchema == nil || decoded.ResponseSchema.Name() != "answer" || llm.IsStrictResponseSchema(decoded.ResponseSchema) {
		t.Errorf("response schema lost: %#v", decoded.ResponseSchema)
	}
	if decoded.Reasoning == nil || decoded.Reasoning.Effort == nil || *decoded.Reasoning.Effort != effort || !decoded.Reasoning.Exclude || decoded.Reasoning.MaxTokens != nil {
		t.Errorf("reasoning lost: %#v", decoded.Reasoning)
	}
	if len(decoded.Modalities) != 2 || decoded.Audio == nil || decoded.Audio.Voice != "alloy" {
		t.Errorf("modalities/audio lost")
	}
	if decoded.SessionID != "s1" {
		t.Errorf("session id lost")
	}
	if decoded.ExtraFields["reasoning_split"] != true || decoded.ExtraFields["n"] != float64(2) {
		t.Errorf("extra fields lost: %#v", decoded.ExtraFields)
	}
}

func TestOptionsDefaultsStayDefaults(t *testing.T) {
	encoded, err := ChatCompletionOptionsToProto(llm.NewChatCompletionOptions(llm.WithMessages(llm.NewMessage(llm.RoleUser, "hi"))))
	if err != nil {
		t.Fatalf("%+v", err)
	}
	funcs, err := ChatCompletionOptionsFromProto(encoded)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	decoded := llm.NewChatCompletionOptions(funcs...)
	if decoded.Temperature != nil || decoded.Seed != nil || decoded.MaxCompletionTokens != nil || decoded.Reasoning != nil || decoded.ResponseSchema != nil {
		t.Errorf("unset options became set: %#v", decoded)
	}
	if decoded.ToolChoice != llm.ToolChoiceAuto || decoded.ResponseFormat != llm.ResponseFormatDefault {
		t.Errorf("defaults changed: %q %q", decoded.ToolChoice, decoded.ResponseFormat)
	}
}

func TestUsageRoundTrip(t *testing.T) {
	cost := 1.25
	original := llm.NewChatCompletionUsageFull(10, 20, 30, 4, 5, &cost, "USD")
	decoded := UsageFromProto(UsageToProto(original))
	if decoded.PromptTokens() != 10 || decoded.CompletionTokens() != 20 || decoded.TotalTokens() != 30 {
		t.Errorf("counters lost")
	}
	if cc := decoded.(llm.CacheCreationReportingUsage); cc.CacheCreationTokens() != 5 {
		t.Errorf("cache creation lost")
	}
	if amount, currency, ok := decoded.(llm.CostReportingUsage).Cost(); !ok || amount != cost || currency != "USD" {
		t.Errorf("cost lost")
	}

	noCost := UsageFromProto(UsageToProto(llm.NewChatCompletionUsage(1, 2, 3)))
	if _, _, ok := noCost.(llm.CostReportingUsage).Cost(); ok {
		t.Errorf("cost appeared from nowhere")
	}
	if UsageFromProto(nil) != nil {
		t.Errorf("nil usage should stay nil")
	}
}

func TestStreamChunkRoundTrip(t *testing.T) {
	usage := llm.NewChatCompletionUsage(1, 2, 3)
	chunks := []llm.StreamChunk{
		llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "hi", llm.NewToolCallDelta(0, "c1", "tool", `{"a"`))),
		llm.NewStreamChunkWithUsage(llm.NewReasoningStreamDelta(llm.RoleAssistant, "", "thinking", nil), usage),
		llm.NewStreamChunk(llm.NewAudioStreamDelta(llm.RoleAssistant, "", "AAAA", "hello")),
		llm.NewCompleteStreamChunk(usage),
		llm.NewErrorStreamChunk(llm.RateLimitError(429, "slow")),
		llm.NewErrorStreamChunkWithUsage(errors.New("boom"), usage),
	}

	for i, original := range chunks {
		decoded, err := StreamChunkFromProto(StreamChunkToProto(original))
		if err != nil {
			t.Fatalf("chunk %d: %+v", i, err)
		}
		if decoded.Type() != original.Type() || decoded.IsComplete() != original.IsComplete() {
			t.Errorf("chunk %d: type %q != %q", i, decoded.Type(), original.Type())
		}
		if (decoded.Usage() == nil) != (original.Usage() == nil) {
			t.Errorf("chunk %d: usage presence changed", i)
		}
		if original.Error() != nil {
			if decoded.Error() == nil || llm.IsRetryable(decoded.Error()) != llm.IsRetryable(original.Error()) {
				t.Errorf("chunk %d: error retryability changed: %v", i, decoded.Error())
			}
		}
		if original.Delta() != nil {
			od, dd := original.Delta(), decoded.Delta()
			if dd.Content() != od.Content() || len(dd.ToolCalls()) != len(od.ToolCalls()) {
				t.Errorf("chunk %d: delta changed", i)
			}
			if len(od.ToolCalls()) > 0 && dd.ToolCalls()[0].ParametersDelta() != od.ToolCalls()[0].ParametersDelta() {
				t.Errorf("chunk %d: tool call delta changed", i)
			}
			if rd, ok := od.(llm.ReasoningStreamDelta); ok && rd.Reasoning() != "" {
				drd, ok := dd.(llm.ReasoningStreamDelta)
				if !ok || drd.Reasoning() != rd.Reasoning() {
					t.Errorf("chunk %d: reasoning lost", i)
				}
			}
			type audioDelta interface {
				AudioData() string
				Transcript() string
			}
			if ad, ok := od.(audioDelta); ok && ad.AudioData() != "" {
				dad, ok := dd.(audioDelta)
				if !ok || dad.AudioData() != ad.AudioData() || dad.Transcript() != ad.Transcript() {
					t.Errorf("chunk %d: audio lost", i)
				}
			}
		}
	}
}

func TestErrorRoundTrip(t *testing.T) {
	cases := []error{
		llm.RateLimitError(429, "slow"),
		llm.RateLimitError(503, "down"),
		llm.NewHTTPError(400, "bad"),
		llm.NewHTTPError(500, "oops"),
		pkgerrors.Wrap(llm.ErrRateLimit, "wrapped"),
		pkgerrors.WithStack(llm.ErrNoMessage),
		pkgerrors.Wrap(llm.ErrUnavailable, "gone"),
		llm.NewValidationError("model", "model is required"),
		pkgerrors.Wrap(context.Canceled, "upstream"),
		pkgerrors.WithStack(context.DeadlineExceeded),
		errors.New("plain"),
	}

	for i, original := range cases {
		decoded := ErrorFromProto(ErrorToProto(original))
		if decoded == nil {
			t.Fatalf("case %d: nil", i)
		}
		if llm.IsRetryable(decoded) != llm.IsRetryable(original) {
			t.Errorf("case %d: retryable %v != %v (%v)", i, llm.IsRetryable(decoded), llm.IsRetryable(original), decoded)
		}
		for _, sentinel := range []error{llm.ErrRateLimit, llm.ErrNoMessage, llm.ErrUnavailable, context.Canceled, context.DeadlineExceeded} {
			if errors.Is(decoded, sentinel) != errors.Is(original, sentinel) {
				t.Errorf("case %d: errors.Is(%v) changed", i, sentinel)
			}
		}
		var oh, dh *llm.HTTPError
		if errors.As(original, &oh) {
			if !errors.As(decoded, &dh) || dh.StatusCode != oh.StatusCode || dh.Body != oh.Body {
				t.Errorf("case %d: http error lost: %v", i, decoded)
			}
		}
		var ov, dv llm.ValidationError
		if errors.As(original, &ov) {
			if !errors.As(decoded, &dv) || dv.Field != ov.Field || dv.Message != ov.Message {
				t.Errorf("case %d: validation error lost: %v", i, decoded)
			}
		}
	}

	wrapped := ErrorFromProto(ErrorToProto(pkgerrors.Wrap(llm.ErrNoMessage, "provider acme")))
	if !errors.Is(wrapped, llm.ErrNoMessage) || wrapped.Error() != "provider acme: no message" {
		t.Errorf("context around the sentinel was lost: %q", wrapped.Error())
	}
	plain := ErrorFromProto(ErrorToProto(llm.NewHTTPError(500, "oops")))
	if plain.Error() != "http 500: oops" {
		t.Errorf("message duplicated: %q", plain.Error())
	}

	if ErrorToProto(nil) != nil || ErrorFromProto(nil) != nil {
		t.Error("nil should stay nil")
	}

	// Through a gRPC status, as unary RPCs carry it.
	decoded := ErrorFromStatus(ErrorToStatus(llm.RateLimitError(429, "slow")))
	if !llm.IsRetryable(decoded) || !errors.Is(decoded, llm.ErrRateLimit) {
		t.Errorf("status round trip lost the error: %v", decoded)
	}

	// Context errors through a status, with and without the typed detail.
	if err := ErrorFromStatus(ErrorToStatus(pkgerrors.Wrap(context.Canceled, "upstream"))); !errors.Is(err, context.Canceled) {
		t.Errorf("cancellation lost through status: %v", err)
	}
	if st, _ := status.FromError(ErrorToStatus(context.DeadlineExceeded)); st.Code() != codes.DeadlineExceeded {
		t.Errorf("deadline should map to codes.DeadlineExceeded, got %v", st.Code())
	}
	if err := ErrorFromStatus(status.Error(codes.Canceled, "context canceled")); !errors.Is(err, context.Canceled) {
		t.Errorf("bare Canceled status should map to context.Canceled, got %v", err)
	}

	// A bare status without typed detail, from a plugin not built with the SDK.
	if err := ErrorFromStatus(status.Error(codes.ResourceExhausted, "429")); !errors.Is(err, llm.ErrRateLimit) || !llm.IsRetryable(err) {
		t.Errorf("ResourceExhausted should map to ErrRateLimit, got %v", err)
	}
	if err := ErrorFromStatus(status.Error(codes.NotFound, "unknown client")); !errors.Is(err, llm.ErrUnavailable) {
		t.Errorf("NotFound should map to ErrUnavailable, got %v", err)
	}
}

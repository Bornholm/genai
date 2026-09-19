package codec

import (
	"github.com/bornholm/genai/llm"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/pkg/errors"
)

var chunkTypeToProto = map[llm.StreamChunkType]pluginv1.StreamChunkType{
	llm.StreamChunkTypeDelta:    pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_DELTA,
	llm.StreamChunkTypeUsage:    pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_USAGE,
	llm.StreamChunkTypeError:    pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_ERROR,
	llm.StreamChunkTypeComplete: pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_COMPLETE,
}

// StreamDeltaToProto converts a delta, probing its optional capabilities.
func StreamDeltaToProto(delta llm.StreamDelta) *pluginv1.StreamDelta {
	if delta == nil {
		return nil
	}
	result := &pluginv1.StreamDelta{
		Role:    RoleToProto(delta.Role()),
		Content: delta.Content(),
	}
	for _, tc := range delta.ToolCalls() {
		result.ToolCalls = append(result.ToolCalls, &pluginv1.ToolCallDelta{
			Index:           int32(tc.Index()),
			Id:              tc.ID(),
			Name:            tc.Name(),
			ParametersDelta: tc.ParametersDelta(),
		})
	}
	if rd, ok := delta.(llm.ReasoningStreamDelta); ok {
		result.Reasoning = rd.Reasoning()
		result.ReasoningDetails = reasoningDetailsToProto(rd.ReasoningDetails())
	}
	type audioDelta interface {
		AudioData() string
		Transcript() string
	}
	if ad, ok := delta.(audioDelta); ok {
		result.AudioData = ad.AudioData()
		result.Transcript = ad.Transcript()
	}
	return result
}

// StreamDeltaFromProto rebuilds a delta.
func StreamDeltaFromProto(delta *pluginv1.StreamDelta) llm.StreamDelta {
	if delta == nil {
		return nil
	}
	role := RoleFromProto(delta.GetRole())
	toolCalls := make([]llm.ToolCallDelta, 0, len(delta.GetToolCalls()))
	for _, tc := range delta.GetToolCalls() {
		toolCalls = append(toolCalls, llm.NewToolCallDelta(int(tc.GetIndex()), tc.GetId(), tc.GetName(), tc.GetParametersDelta()))
	}
	switch {
	case delta.GetAudioData() != "" || delta.GetTranscript() != "":
		return llm.NewAudioStreamDelta(role, delta.GetContent(), delta.GetAudioData(), delta.GetTranscript(), toolCalls...)
	case delta.GetReasoning() != "" || len(delta.GetReasoningDetails()) > 0:
		return llm.NewReasoningStreamDelta(role, delta.GetContent(), delta.GetReasoning(), reasoningDetailsFromProto(delta.GetReasoningDetails()), toolCalls...)
	default:
		return llm.NewStreamDelta(role, delta.GetContent(), toolCalls...)
	}
}

// StreamChunkToProto converts a chunk.
func StreamChunkToProto(chunk llm.StreamChunk) *pluginv1.StreamChunk {
	result := &pluginv1.StreamChunk{
		Type:  chunkTypeToProto[chunk.Type()],
		Delta: StreamDeltaToProto(chunk.Delta()),
		Usage: UsageToProto(chunk.Usage()),
	}
	if err := chunk.Error(); err != nil {
		result.Error = ErrorToProto(err)
	}
	if chunk.IsComplete() && result.Type == pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_UNSPECIFIED {
		result.Type = pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_COMPLETE
	}
	return result
}

// StreamChunkFromProto rebuilds a chunk with the llm constructor matching its
// type.
func StreamChunkFromProto(chunk *pluginv1.StreamChunk) (llm.StreamChunk, error) {
	usage := UsageFromProto(chunk.GetUsage())
	switch chunk.GetType() {
	case pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_ERROR:
		err := ErrorFromProto(chunk.GetError())
		if err == nil {
			err = errors.New("plugin reported an error without details")
		}
		if usage != nil {
			return llm.NewErrorStreamChunkWithUsage(err, usage), nil
		}
		return llm.NewErrorStreamChunk(err), nil

	case pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_COMPLETE:
		if usage == nil {
			usage = llm.NewChatCompletionUsage(0, 0, 0)
		}
		return llm.NewCompleteStreamChunk(usage), nil

	case pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_DELTA, pluginv1.StreamChunkType_STREAM_CHUNK_TYPE_USAGE:
		delta := StreamDeltaFromProto(chunk.GetDelta())
		if delta == nil {
			delta = llm.NewStreamDelta(llm.RoleAssistant, "")
		}
		if usage != nil {
			return llm.NewStreamChunkWithUsage(delta, usage), nil
		}
		return llm.NewStreamChunk(delta), nil

	default:
		return nil, errors.Errorf("unknown stream chunk type %q", chunk.GetType().String())
	}
}

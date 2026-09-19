package codec

import (
	"github.com/bornholm/genai/llm"
	pluginv1 "github.com/bornholm/genai/llm/provider/plugin/proto/genai/plugin/v1"
	"github.com/pkg/errors"
)

// UsageToProto converts a usage, probing its optional capabilities.
func UsageToProto(usage llm.ChatCompletionUsage) *pluginv1.Usage {
	if usage == nil {
		return nil
	}
	result := &pluginv1.Usage{
		PromptTokens:     usage.PromptTokens(),
		CompletionTokens: usage.CompletionTokens(),
		TotalTokens:      usage.TotalTokens(),
	}
	type cachedUsage interface{ CachedTokens() int64 }
	if cu, ok := usage.(cachedUsage); ok {
		result.CachedTokens = cu.CachedTokens()
	}
	if cc, ok := usage.(llm.CacheCreationReportingUsage); ok {
		result.CacheCreationTokens = cc.CacheCreationTokens()
	}
	if cr, ok := usage.(llm.CostReportingUsage); ok {
		if amount, currency, ok := cr.Cost(); ok {
			result.Cost = &amount
			result.CostCurrency = currency
		}
	}
	return result
}

// UsageFromProto rebuilds a usage. A nil proto gives a nil usage.
func UsageFromProto(usage *pluginv1.Usage) llm.ChatCompletionUsage {
	if usage == nil {
		return nil
	}
	return llm.NewChatCompletionUsageFull(
		usage.GetPromptTokens(),
		usage.GetCompletionTokens(),
		usage.GetTotalTokens(),
		usage.GetCachedTokens(),
		usage.GetCacheCreationTokens(),
		usage.Cost,
		usage.GetCostCurrency(),
	)
}

// ChatCompletionResponseToProto converts a response.
func ChatCompletionResponseToProto(res llm.ChatCompletionResponse) (*pluginv1.ChatCompletionResponse, error) {
	message, err := MessageToProto(res.Message())
	if err != nil {
		return nil, errors.WithStack(err)
	}
	toolCalls, err := toolCallsToProto(res.ToolCalls())
	if err != nil {
		return nil, errors.WithStack(err)
	}
	result := &pluginv1.ChatCompletionResponse{
		Message:   message,
		ToolCalls: toolCalls,
		Usage:     UsageToProto(res.Usage()),
	}
	if rr, ok := res.(llm.ReasoningChatCompletionResponse); ok {
		result.Reasoning = rr.Reasoning()
		result.ReasoningDetails = reasoningDetailsToProto(rr.ReasoningDetails())
	}
	return result, nil
}

// ChatCompletionResponseFromProto rebuilds a response.
func ChatCompletionResponseFromProto(res *pluginv1.ChatCompletionResponse) (llm.ChatCompletionResponse, error) {
	if res.GetMessage() == nil {
		return nil, errors.New("response carries no message")
	}
	message, err := MessageFromProto(res.GetMessage())
	if err != nil {
		return nil, errors.WithStack(err)
	}
	usage := UsageFromProto(res.GetUsage())
	if usage == nil {
		usage = llm.NewChatCompletionUsage(0, 0, 0)
	}
	toolCalls := toolCallsFromProto(res.GetToolCalls())
	if res.GetReasoning() != "" || len(res.GetReasoningDetails()) > 0 {
		return llm.NewChatCompletionResponseWithReasoning(message, usage, res.GetReasoning(), reasoningDetailsFromProto(res.GetReasoningDetails()), toolCalls...), nil
	}
	return llm.NewChatCompletionResponse(message, usage, toolCalls...), nil
}

// EmbeddingsResponse is the llm.EmbeddingsResponse rebuilt on the host side.
type EmbeddingsResponse struct {
	embeddings [][]float64
	usage      llm.EmbeddingsUsage
}

// Embeddings implements llm.EmbeddingsResponse.
func (r *EmbeddingsResponse) Embeddings() [][]float64 { return r.embeddings }

// Usage implements llm.EmbeddingsResponse.
func (r *EmbeddingsResponse) Usage() llm.EmbeddingsUsage { return r.usage }

var _ llm.EmbeddingsResponse = &EmbeddingsResponse{}

// EmbeddingsResponseToProto converts an embeddings response.
func EmbeddingsResponseToProto(res llm.EmbeddingsResponse) *pluginv1.EmbeddingsResponse {
	embeddings := make([]*pluginv1.Embedding, 0, len(res.Embeddings()))
	for _, e := range res.Embeddings() {
		embeddings = append(embeddings, &pluginv1.Embedding{Values: e})
	}
	result := &pluginv1.EmbeddingsResponse{Embeddings: embeddings}
	if usage := res.Usage(); usage != nil {
		result.Usage = &pluginv1.EmbeddingsUsage{
			PromptTokens: usage.PromptTokens(),
			TotalTokens:  usage.TotalTokens(),
		}
	}
	return result
}

// EmbeddingsResponseFromProto rebuilds an embeddings response.
func EmbeddingsResponseFromProto(res *pluginv1.EmbeddingsResponse) *EmbeddingsResponse {
	embeddings := make([][]float64, 0, len(res.GetEmbeddings()))
	for _, e := range res.GetEmbeddings() {
		embeddings = append(embeddings, e.GetValues())
	}
	return &EmbeddingsResponse{
		embeddings: embeddings,
		usage:      llm.NewEmbeddingsUsage(res.GetUsage().GetPromptTokens(), res.GetUsage().GetTotalTokens()),
	}
}

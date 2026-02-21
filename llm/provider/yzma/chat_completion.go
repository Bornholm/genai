package yzma

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/bornholm/genai/llm"
	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
	"github.com/hybridgroup/yzma/pkg/template"
	"github.com/pkg/errors"
)

type ChatCompletionClient struct {
	modelPath       string
	modelURL        string
	libPath         string
	processor       string
	version         string
	contextSize     int
	batchSize       int
	uBatchSize      int
	temperature     float64
	topK            int
	topP            float64
	minP            float64
	presencePenalty float64
	penaltyLastN    int
	predictSize     int
	template        string
	verbose         bool

	// Runtime state
	mu     sync.Mutex
	model  llama.Model
	vocab  llama.Vocab
	lctx   llama.Context
	loaded bool
}

// ChatCompletion implements llm.ChatCompletionClient.
func (c *ChatCompletionClient) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	// Validate options before proceeding
	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	// Ensure the model is loaded
	if err := c.ensureLoaded(); err != nil {
		return nil, errors.WithStack(err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Clear KV cache before starting new generation
	if err := c.clearMemory(); err != nil {
		return nil, errors.WithStack(err)
	}

	// Build prompt with tools support
	prompt, err := c.buildPrompt(opts)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	// Tokenize
	tokens := llama.Tokenize(c.vocab, prompt, true, true)

	// Decode prompt tokens in batches to handle large prompts
	if err := c.decodePromptTokens(tokens); err != nil {
		return nil, errors.WithStack(err)
	}

	// Create sampler
	sp := llama.DefaultSamplerParams()
	sp.Temp = float32(opts.Temperature)
	sp.TopK = int32(c.topK)
	sp.TopP = float32(c.topP)
	sp.MinP = float32(c.minP)
	sp.PenaltyPresent = float32(c.presencePenalty)
	sp.PenaltyLastN = int32(c.penaltyLastN)

	samplers := []llama.SamplerType{
		llama.SamplerTypePenalties,
		llama.SamplerTypeTopK,
		llama.SamplerTypeTopP,
		llama.SamplerTypeMinP,
		llama.SamplerTypeTemperature,
	}
	sampler := llama.NewSampler(c.model, samplers, sp)

	// Generate response
	response := ""
	maxTokens := c.predictSize
	if opts.MaxCompletionTokens != nil {
		maxTokens = *opts.MaxCompletionTokens
	}

	for pos := int32(0); pos < int32(maxTokens); pos++ {
		select {
		case <-ctx.Done():
			return nil, errors.WithStack(ctx.Err())
		default:
		}

		token := llama.SamplerSample(sampler, c.lctx, -1)

		if llama.VocabIsEOG(c.vocab, token) {
			break
		}

		tokenBuf := make([]byte, 256)
		l := llama.TokenToPiece(c.vocab, token, tokenBuf, 0, false)
		response += string(tokenBuf[:l])

		// Decode the generated token
		batch := llama.BatchGetOne([]llama.Token{token})
		if _, err := llama.Decode(c.lctx, batch); err != nil {
			return nil, errors.Wrap(err, "failed to decode token")
		}
	}

	// Strip think blocks before parsing tool calls so we only see real tool calls
	responseWithoutThink := stripThinkBlocks(response)

	// Parse tool calls from response
	toolCalls := c.parseToolCalls(responseWithoutThink)

	// Clean response content (remove tool call markers for the message content)
	cleanResponse := c.cleanResponse(response)

	// Create response message
	message := llm.NewMessage(llm.RoleAssistant, cleanResponse)

	// Note: yzma doesn't provide token usage directly, so we estimate
	promptTokens := int64(len(tokens))
	completionTokens := int64(len(response) / 4) // Rough estimate
	totalTokens := promptTokens + completionTokens

	usage := llm.NewChatCompletionUsage(promptTokens, completionTokens, totalTokens)

	return llm.NewChatCompletionResponse(message, usage, toolCalls...), nil
}

// ChatCompletionStream implements llm.ChatCompletionStreamingClient.
func (c *ChatCompletionClient) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	opts := llm.NewChatCompletionOptions(funcs...)

	// Validate options before proceeding
	if err := opts.Validate(); err != nil {
		return nil, errors.WithStack(err)
	}

	// Ensure the model is loaded
	if err := c.ensureLoaded(); err != nil {
		return nil, errors.WithStack(err)
	}

	chunks := make(chan llm.StreamChunk, 10)

	go func() {
		defer close(chunks)

		c.mu.Lock()
		defer c.mu.Unlock()

		// Clear KV cache before starting new generation
		if err := c.clearMemory(); err != nil {
			chunks <- llm.NewErrorStreamChunk(errors.WithStack(err))
			return
		}

		// Build prompt with tools support
		prompt, err := c.buildPrompt(opts)
		if err != nil {
			chunks <- llm.NewErrorStreamChunk(errors.WithStack(err))
			return
		}

		// Tokenize
		tokens := llama.Tokenize(c.vocab, prompt, true, true)

		// Decode prompt tokens in batches to handle large prompts
		if err := c.decodePromptTokens(tokens); err != nil {
			chunks <- llm.NewErrorStreamChunk(errors.WithStack(err))
			return
		}

		// Create sampler
		sp := llama.DefaultSamplerParams()
		sp.Temp = float32(opts.Temperature)
		sp.TopK = int32(c.topK)
		sp.TopP = float32(c.topP)
		sp.MinP = float32(c.minP)
		sp.PenaltyPresent = float32(c.presencePenalty)
		sp.PenaltyLastN = int32(c.penaltyLastN)

		samplers := []llama.SamplerType{
			llama.SamplerTypePenalties,
			llama.SamplerTypeTopK,
			llama.SamplerTypeTopP,
			llama.SamplerTypeMinP,
			llama.SamplerTypeTemperature,
		}
		sampler := llama.NewSampler(c.model, samplers, sp)

		// Generate response with streaming
		maxTokens := c.predictSize
		if opts.MaxCompletionTokens != nil {
			maxTokens = *opts.MaxCompletionTokens
		}

		var promptTokens, completionTokens int64

		// Buffer to detect and suppress <think>...</think> blocks in the stream.
		var thinkBuf string
		inThink := false

		for pos := int32(0); pos < int32(maxTokens); pos++ {
			select {
			case <-ctx.Done():
				chunks <- llm.NewErrorStreamChunk(errors.WithStack(ctx.Err()))
				return
			default:
			}

			token := llama.SamplerSample(sampler, c.lctx, -1)

			if llama.VocabIsEOG(c.vocab, token) {
				break
			}

			tokenBuf := make([]byte, 256)
			l := llama.TokenToPiece(c.vocab, token, tokenBuf, 0, false)
			content := string(tokenBuf[:l])
			completionTokens++

			// Suppress think blocks from the stream.
			if inThink {
				thinkBuf += content
				if idx := strings.Index(thinkBuf, "</think>"); idx != -1 {
					inThink = false
					thinkBuf = ""
				}
			} else {
				pending := content
				if idx := strings.Index(pending, "<think>"); idx != -1 {
					// Emit any content before the think block, then buffer the rest.
					before := pending[:idx]
					if before != "" {
						chunks <- llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, before))
					}
					inThink = true
					thinkBuf = pending[idx:]
					// Check if think block closes in the same chunk.
					if closeIdx := strings.Index(thinkBuf, "</think>"); closeIdx != -1 {
						inThink = false
						thinkBuf = ""
					}
				} else {
					chunks <- llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, pending))
				}
			}

			// Decode the generated token
			batch := llama.BatchGetOne([]llama.Token{token})
			if _, err := llama.Decode(c.lctx, batch); err != nil {
				chunks <- llm.NewErrorStreamChunk(errors.Wrap(err, "failed to decode token"))
				return
			}
		}

		// Send completion chunk
		promptTokens = int64(len(tokens))
		usage := llm.NewChatCompletionUsage(promptTokens, completionTokens, promptTokens+completionTokens)
		chunks <- llm.NewCompleteStreamChunk(usage)
	}()

	return chunks, nil
}

// ensureLoaded loads the model if not already loaded
func (c *ChatCompletionClient) ensureLoaded() error {
	if c.loaded {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.loaded {
		return nil
	}

	// Load the library
	if err := llama.Load(c.libPath); err != nil {
		return errors.Wrap(err, "unable to load llama library")
	}

	if !c.verbose {
		llama.LogSet(llama.LogSilent())
	}

	llama.Init()

	// Load the model
	mParams := llama.ModelDefaultParams()
	model, err := llama.ModelLoadFromFile(c.modelPath, mParams)
	if err != nil {
		return errors.Wrap(err, "unable to load model from file")
	}
	if model == 0 {
		return errors.Errorf("unable to load model from file: %s", c.modelPath)
	}

	c.model = model
	c.vocab = llama.ModelGetVocab(model)

	// Create context
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(c.contextSize)
	ctxParams.NBatch = uint32(c.batchSize)
	ctxParams.NUbatch = uint32(c.uBatchSize)

	lctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		llama.ModelFree(model)
		return errors.Wrap(err, "unable to initialize context from model")
	}

	c.lctx = lctx
	c.loaded = true

	return nil
}

// clearMemory clears the KV cache memory
func (c *ChatCompletionClient) clearMemory() error {
	if c.lctx == 0 {
		return nil
	}

	mem, err := llama.GetMemory(c.lctx)
	if err != nil {
		return errors.Wrap(err, "failed to get memory")
	}

	if mem != 0 {
		if err := llama.MemoryClear(mem, true); err != nil {
			return errors.Wrap(err, "failed to clear memory")
		}
	}

	return nil
}

// decodePromptTokens decodes prompt tokens in batches to handle large prompts
func (c *ChatCompletionClient) decodePromptTokens(tokens []llama.Token) error {
	// Handle encoder models
	if llama.ModelHasEncoder(c.model) {
		batch := llama.BatchGetOne(tokens)
		llama.Encode(c.lctx, batch)
		start := llama.ModelDecoderStartToken(c.model)
		if start == llama.TokenNull {
			start = llama.VocabBOS(c.vocab)
		}
		batch = llama.BatchGetOne([]llama.Token{start})
		_, err := llama.Decode(c.lctx, batch)
		return err
	}

	// Split tokens into chunks that fit within batch size
	batchSize := int32(c.batchSize)
	for i := int32(0); i < int32(len(tokens)); i += batchSize {
		end := i + batchSize
		if end > int32(len(tokens)) {
			end = int32(len(tokens))
		}

		chunk := tokens[i:end]
		batch := llama.BatchGetOne(chunk)

		if _, err := llama.Decode(c.lctx, batch); err != nil {
			return errors.Wrapf(err, "failed to decode prompt tokens at position %d", i)
		}
	}

	return nil
}

// Close releases the model resources
func (c *ChatCompletionClient) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.loaded {
		return
	}

	if c.lctx != 0 {
		llama.Free(c.lctx)
		c.lctx = 0
	}

	if c.model != 0 {
		llama.ModelFree(c.model)
		c.model = 0
	}

	c.loaded = false
}

// messageData holds Go string representation of messages for manipulation
type messageData struct {
	role    string
	content string
}

// buildPrompt builds the prompt with tools support using yzma template package
func (c *ChatCompletionClient) buildPrompt(opts *llm.ChatCompletionOptions) (string, error) {
	// Convert messages to yzma message format
	messages := c.convertToYzmaMessages(opts.Messages)

	// Get chat template from model
	chatTemplate := c.template
	if chatTemplate == "" {
		chatTemplate = llama.ModelChatTemplate(c.model, "")
	}

	// If JSON response format is requested, inject instruction into system prompt
	if opts.ResponseFormat == llm.ResponseFormatJSON {
		jsonInstruction := "You must respond with valid JSON only. Do not include any text outside of the JSON object."
		hasSystem := false
		for i, msg := range messages {
			if chatMsg, ok := msg.(message.Chat); ok && chatMsg.Role == "system" {
				messages[i] = message.Chat{
					Role:    "system",
					Content: chatMsg.Content + "\n\n" + jsonInstruction,
				}
				hasSystem = true
				break
			}
		}
		if !hasSystem {
			messages = append([]message.Message{
				message.Chat{Role: "system", Content: jsonInstruction},
			}, messages...)
		}
	}

	// If we have tools, inject tool definitions into the system prompt
	if len(opts.Tools) > 0 {
		toolsJSON, err := json.MarshalIndent(c.convertTools(opts.Tools), "", "  ")
		if err != nil {
			return "", errors.Wrap(err, "failed to marshal tools")
		}

		// Build system prompt with tool instructions (using yzma multitool format)
		systemPrompt := fmt.Sprintf(`You are a helpful assistant with access to the following tools:

%s

When you need to use a tool, respond with a tool call in the following format:
<tool_call>
{"name": "function_name", "arguments": {"arg1": "value1", "arg2": "value2"}}
</tool_call>

You can make multiple tool calls to solve complex problems step by step.
After receiving all tool results, provide a final answer to the user.
Do not include tool calls in your final answer.`, string(toolsJSON))

		// Check if there's already a system message
		hasSystem := false
		for i, msg := range messages {
			if chatMsg, ok := msg.(message.Chat); ok && chatMsg.Role == "system" {
				// Prepend tool instructions to existing system message
				messages[i] = message.Chat{
					Role:    "system",
					Content: systemPrompt + "\n\n" + chatMsg.Content,
				}
				hasSystem = true
				break
			}
		}

		if !hasSystem {
			// Add system message at the beginning
			messages = append([]message.Message{
				message.Chat{Role: "system", Content: systemPrompt},
			}, messages...)
		}
	}

	// Try to apply template using yzma template package first
	prompt, err := template.Apply(chatTemplate, messages, true)
	if err == nil {
		return prompt, nil
	}

	// If yzma template fails (e.g., complex Jinja2 syntax not supported by gonja),
	// fall back to llama.ChatApplyTemplate which uses the native llama.cpp template handler
	llamaMessages := c.convertToLlamaMessages(messages)
	buf := make([]byte, 32768) // Larger buffer for complex templates
	promptLen := llama.ChatApplyTemplate(chatTemplate, llamaMessages, true, buf)
	if promptLen <= 0 {
		return "", errors.New("failed to apply chat template")
	}

	return string(buf[:promptLen]), nil
}

// convertToLlamaMessages converts yzma message.Message slice to llama.ChatMessage slice
func (c *ChatCompletionClient) convertToLlamaMessages(messages []message.Message) []llama.ChatMessage {
	result := make([]llama.ChatMessage, 0, len(messages))

	for _, msg := range messages {
		switch m := msg.(type) {
		case message.Chat:
			result = append(result, llama.NewChatMessage(m.Role, m.Content))
		case message.Tool:
			// Convert tool calls to text representation
			var toolCallStrs []string
			for _, tc := range m.ToolCalls {
				argsJSON, _ := json.Marshal(tc.Function.Arguments)
				toolCallStrs = append(toolCallStrs, fmt.Sprintf(`brie
{"name": "%s", "arguments": %s}
eieb`, tc.Function.Name, string(argsJSON)))
			}
			content := strings.Join(toolCallStrs, "\n")
			result = append(result, llama.NewChatMessage("assistant", content))
		case message.ToolResponse:
			result = append(result, llama.NewChatMessage("tool", m.Content))
		}
	}

	return result
}

// convertToYzmaMessages converts llm.Message slice to yzma message.Message slice
func (c *ChatCompletionClient) convertToYzmaMessages(messages []llm.Message) []message.Message {
	result := make([]message.Message, 0, len(messages))

	for _, msg := range messages {
		role := string(msg.Role())
		content := msg.Content()

		// Handle tool calls message
		if tcMsg, ok := msg.(llm.ToolCallsMessage); ok {
			// Convert tool calls to yzma ToolCall format
			toolCalls := make([]message.ToolCall, len(tcMsg.ToolCalls()))
			for i, tc := range tcMsg.ToolCalls() {
				// Parse arguments
				var args map[string]any
				switch p := tc.Parameters().(type) {
				case string:
					json.Unmarshal([]byte(p), &args)
				case map[string]any:
					args = p
				}

				// Convert to string map for yzma
				strArgs := make(map[string]string)
				for k, v := range args {
					strArgs[k] = fmt.Sprintf("%v", v)
				}

				toolCalls[i] = message.ToolCall{
					Type: "function",
					Function: message.ToolFunction{
						Name:      tc.Name(),
						Arguments: strArgs,
					},
				}
			}

			result = append(result, message.Tool{
				Role:      "assistant",
				ToolCalls: toolCalls,
			})
			continue
		}

		// Handle tool message
		if msg.Role() == llm.RoleTool {
			result = append(result, message.ToolResponse{
				Role:    "tool",
				Content: content,
			})
			continue
		}

		// Strip think blocks from historical assistant messages (Qwen3 best practice:
		// thinking content should not be included in multi-turn history).
		if msg.Role() == llm.RoleAssistant {
			content = stripThinkBlocks(content)
		}

		result = append(result, message.Chat{
			Role:    role,
			Content: content,
		})
	}

	return result
}

// convertTools converts llm.Tool slice to yzma tool definitions
func (c *ChatCompletionClient) convertTools(tools []llm.Tool) []map[string]any {
	result := make([]map[string]any, 0, len(tools))

	for _, tool := range tools {
		result = append(result, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        tool.Name(),
				"description": tool.Description(),
				"parameters":  tool.Parameters(),
			},
		})
	}

	return result
}

// parseToolCalls parses tool calls from the model response
func (c *ChatCompletionClient) parseToolCalls(response string) []llm.ToolCall {
	toolCalls := message.ParseToolCalls(response)

	result := make([]llm.ToolCall, 0, len(toolCalls))
	for i, tc := range toolCalls {
		// Convert string arguments to proper types
		// The yzma message.ParseToolCalls returns map[string]string,
		// but we need to preserve the original JSON types
		args := make(map[string]any)
		for k, v := range tc.Function.Arguments {
			// Try to parse as JSON value to preserve types
			var parsed any
			if err := json.Unmarshal([]byte(v), &parsed); err == nil {
				args[k] = parsed
			} else {
				// Keep as string if not valid JSON
				args[k] = v
			}
		}

		// Convert arguments to JSON string
		argsJSON, err := json.Marshal(args)
		if err != nil {
			continue
		}

		result = append(result, llm.NewToolCall(
			fmt.Sprintf("call_%d", i),
			tc.Function.Name,
			string(argsJSON),
		))
	}

	return result
}

// stripThinkBlocks removes <think>...</think> reasoning blocks from a response.
func stripThinkBlocks(s string) string {
	for {
		start := strings.Index(s, "<think>")
		if start == -1 {
			break
		}
		end := strings.Index(s, "</think>")
		if end == -1 {
			s = s[:start]
			break
		}
		s = s[:start] + s[end+len("</think>"):]
	}
	return s
}

// cleanResponse removes think blocks and tool call markers from the response.
func (c *ChatCompletionClient) cleanResponse(response string) string {
	response = stripThinkBlocks(response)
	// Remove <tool_call> tags and their content
	for {
		start := strings.Index(response, "<tool_call>")
		if start == -1 {
			break
		}
		end := strings.Index(response, "</tool_call>")
		if end == -1 || end < start {
			break
		}
		response = response[:start] + response[end+len("</tool_call>"):]
	}
	return strings.TrimSpace(response)
}

// OptionFunc is a function that configures the ChatCompletionClient
type OptionFunc func(c *ChatCompletionClient) error

// WithModelPath sets the path to the GGUF model file
func WithModelPath(path string) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.modelPath = path
		return nil
	}
}

// WithModelURL sets a URL to download the GGUF model file from if not already present locally.
// Supports Hugging Face URLs and any URL supported by go-getter.
func WithModelURL(url string) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.modelURL = url
		return nil
	}
}

// WithLibPath sets the path to the llama.cpp library directory
func WithLibPath(path string) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.libPath = path
		return nil
	}
}

// WithProcessor sets the processor type (cpu, cuda, vulkan, metal)
func WithProcessor(processor string) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.processor = processor
		return nil
	}
}

// WithVersion sets the llama.cpp version to download
func WithVersion(version string) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.version = version
		return nil
	}
}

// WithContextSize sets the context size
func WithContextSize(size int) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.contextSize = size
		return nil
	}
}

// WithBatchSize sets the batch size
func WithBatchSize(size int) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.batchSize = size
		return nil
	}
}

// WithUBatchSize sets the micro-batch size
func WithUBatchSize(size int) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.uBatchSize = size
		return nil
	}
}

// WithTemperature sets the sampling temperature
func WithTemperature(temp float64) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.temperature = temp
		return nil
	}
}

// WithTopK sets the top-k sampling parameter
func WithTopK(k int) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.topK = k
		return nil
	}
}

// WithTopP sets the top-p sampling parameter
func WithTopP(p float64) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.topP = p
		return nil
	}
}

// WithMinP sets the min-p sampling parameter
func WithMinP(p float64) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.minP = p
		return nil
	}
}

// WithPresencePenalty sets the presence penalty (0.0 = disabled, recommended 2.0 for non-thinking mode).
func WithPresencePenalty(p float64) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.presencePenalty = p
		return nil
	}
}

// WithPenaltyLastN sets how many previous tokens are considered for penalties (-1 = context size).
func WithPenaltyLastN(n int) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.penaltyLastN = n
		return nil
	}
}

// WithPredictSize sets the maximum number of tokens to predict
func WithPredictSize(size int) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.predictSize = size
		return nil
	}
}

// WithTemplate sets the chat template
func WithTemplate(template string) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.template = template
		return nil
	}
}

// WithVerbose enables verbose logging
func WithVerbose(verbose bool) OptionFunc {
	return func(c *ChatCompletionClient) error {
		c.verbose = verbose
		return nil
	}
}

// NewChatCompletionClient creates a new ChatCompletionClient
func NewChatCompletionClient(funcs ...OptionFunc) (*ChatCompletionClient, error) {
	client := &ChatCompletionClient{
		contextSize:     40960,
		batchSize:       512,
		uBatchSize:      512,
		temperature:     1.0,
		topK:            20,
		topP:            1.0,
		minP:            0.0,
		presencePenalty: 2.0,
		penaltyLastN:    64,
		predictSize: 32768,
		verbose:     false,
	}

	for _, fn := range funcs {
		if err := fn(client); err != nil {
			return nil, errors.WithStack(err)
		}
	}

	// Auto-download model if needed
	if err := client.ensureModel(); err != nil {
		return nil, errors.WithStack(err)
	}

	// Auto-download binaries if needed
	if err := client.ensureBinaries(); err != nil {
		return nil, errors.WithStack(err)
	}

	return client, nil
}

// ensureModel downloads the model file from modelURL if modelPath is not set or does not exist.
func (c *ChatCompletionClient) ensureModel() error {
	if c.modelURL == "" {
		return nil
	}

	// Strip query params to get clean filename
	cleanURL := strings.SplitN(c.modelURL, "?", 2)[0]
	filename := filepath.Base(cleanURL)

	// Derive local path from URL filename if not set
	if c.modelPath == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return errors.Wrap(err, "failed to get user home directory")
		}
		c.modelPath = filepath.Join(homeDir, ".yzma", "models", filename)
	}

	// Download only if the file doesn't already exist
	if _, err := os.Stat(c.modelPath); err == nil {
		return nil
	}

	modelsDir := filepath.Dir(c.modelPath)
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		return errors.Wrap(err, "failed to create model directory")
	}

	if err := download.GetModel(c.modelURL, modelsDir); err != nil {
		return errors.Wrap(err, "failed to download model")
	}

	return nil
}

// ensureBinaries downloads llama.cpp binaries if not already present
func (c *ChatCompletionClient) ensureBinaries() error {
	// If lib path is already set and exists, use it
	if c.libPath != "" {
		if _, err := os.Stat(c.libPath); err == nil {
			return nil
		}
	}

	// Set default lib path
	if c.libPath == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return errors.Wrap(err, "failed to get user home directory")
		}
		c.libPath = filepath.Join(homeDir, ".yzma", "lib")
	}

	// Check if already installed
	if download.AlreadyInstalled(c.libPath) {
		return nil
	}

	// Get version
	version := c.version
	if version == "" {
		var err error
		version, err = download.LlamaLatestVersion()
		if err != nil {
			return errors.Wrap(err, "could not obtain latest llama.cpp version")
		}
	}

	// Determine processor type
	processor := c.processor
	if processor == "" {
		processor = "cpu"
		if cudaInstalled, _ := download.HasCUDA(); cudaInstalled {
			processor = "cuda"
		}
	}

	// Download binaries
	if err := download.Get(runtime.GOARCH, runtime.GOOS, processor, version, c.libPath); err != nil {
		return errors.Wrap(err, "failed to download llama.cpp binaries")
	}

	return nil
}

var _ llm.ChatCompletionClient = &ChatCompletionClient{}
var _ llm.ChatCompletionStreamingClient = &ChatCompletionClient{}

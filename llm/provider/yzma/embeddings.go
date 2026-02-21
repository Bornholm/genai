package yzma

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/bornholm/genai/llm"
	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/pkg/errors"
)

type EmbeddingsClient struct {
	modelPath   string
	modelURL    string
	libPath     string
	processor   string
	version     string
	contextSize int
	batchSize   int
	poolingType llama.PoolingType
	normalize   bool
	verbose     bool

	// Runtime state
	mu        sync.Mutex
	model     llama.Model
	vocab     llama.Vocab
	lctx      llama.Context
	nEmbd     int32
	loaded    bool
	libLoaded bool
}

// Embeddings implements llm.EmbeddingsClient.
func (c *EmbeddingsClient) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	opts := llm.NewEmbeddingsOptions(funcs...)

	// Ensure the model is loaded
	if err := c.ensureLoaded(); err != nil {
		return nil, errors.WithStack(err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	embeddings := make([][]float64, 0, len(inputs))
	var totalTokens int64

	for _, input := range inputs {
		select {
		case <-ctx.Done():
			return nil, errors.WithStack(ctx.Err())
		default:
		}

		// Tokenize input
		tokens := llama.Tokenize(c.vocab, input, true, true)
		totalTokens += int64(len(tokens))

		// Create batch and decode
		batch := llama.BatchGetOne(tokens)
		if _, err := llama.Decode(c.lctx, batch); err != nil {
			return nil, errors.Wrapf(err, "failed to decode input: %s", input)
		}

		// Get embeddings
		vec, err := llama.GetEmbeddingsSeq(c.lctx, 0, c.nEmbd)
		if err != nil {
			return nil, errors.Wrap(err, "unable to get embeddings")
		}

		// Convert to float64 and optionally normalize
		embedding := make([]float64, len(vec))
		if c.normalize {
			// Calculate norm
			var sum float64
			for _, v := range vec {
				sum += float64(v * v)
			}
			norm := math.Sqrt(sum)
			invNorm := 1.0 / norm

			// Normalize and convert
			for i, v := range vec {
				embedding[i] = float64(v) * invNorm
			}
		} else {
			for i, v := range vec {
				embedding[i] = float64(v)
			}
		}

		embeddings = append(embeddings, embedding)
	}

	// Apply dimensions if specified
	if opts.Dimensions != nil && *opts.Dimensions > 0 {
		embeddings = c.truncateEmbeddings(embeddings, *opts.Dimensions)
	}

	return &BaseEmbeddingsResponse{
		embeddings: embeddings,
		usage:      llm.NewEmbeddingsUsage(totalTokens, totalTokens),
	}, nil
}

// truncateEmbeddings truncates embeddings to the specified dimensions
func (c *EmbeddingsClient) truncateEmbeddings(embeddings [][]float64, dimensions int) [][]float64 {
	result := make([][]float64, len(embeddings))
	for i, emb := range embeddings {
		if len(emb) > dimensions {
			result[i] = emb[:dimensions]
		} else {
			result[i] = emb
		}
	}
	return result
}

// ensureLoaded loads the model if not already loaded
func (c *EmbeddingsClient) ensureLoaded() error {
	if c.loaded {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.loaded {
		return nil
	}

	// Load the library (only once)
	if !c.libLoaded {
		if err := llama.Load(c.libPath); err != nil {
			return errors.Wrap(err, "unable to load llama library")
		}
		c.libLoaded = true
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
	c.nEmbd = llama.ModelNEmbd(model)

	// Create context with embeddings enabled
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(c.contextSize)
	ctxParams.NBatch = uint32(c.batchSize)
	ctxParams.PoolingType = c.poolingType
	ctxParams.Embeddings = 1

	lctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		llama.ModelFree(model)
		return errors.Wrap(err, "unable to initialize context from model")
	}

	c.lctx = lctx
	c.loaded = true

	return nil
}

// Close releases the model resources
func (c *EmbeddingsClient) Close() {
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

// EmbeddingsOptionFunc is a function that configures the EmbeddingsClient
type EmbeddingsOptionFunc func(c *EmbeddingsClient) error

// WithEmbeddingsModelPath sets the path to the GGUF model file
func WithEmbeddingsModelPath(path string) EmbeddingsOptionFunc {
	return func(c *EmbeddingsClient) error {
		c.modelPath = path
		return nil
	}
}

// WithEmbeddingsModelURL sets a URL to download the GGUF model file from if not already present locally.
// Supports Hugging Face URLs and any URL supported by go-getter.
func WithEmbeddingsModelURL(url string) EmbeddingsOptionFunc {
	return func(c *EmbeddingsClient) error {
		c.modelURL = url
		return nil
	}
}

// WithEmbeddingsLibPath sets the path to the llama.cpp library directory
func WithEmbeddingsLibPath(path string) EmbeddingsOptionFunc {
	return func(c *EmbeddingsClient) error {
		c.libPath = path
		return nil
	}
}

// WithEmbeddingsProcessor sets the processor type (cpu, cuda, vulkan, metal)
func WithEmbeddingsProcessor(processor string) EmbeddingsOptionFunc {
	return func(c *EmbeddingsClient) error {
		c.processor = processor
		return nil
	}
}

// WithEmbeddingsVersion sets the llama.cpp version to download
func WithEmbeddingsVersion(version string) EmbeddingsOptionFunc {
	return func(c *EmbeddingsClient) error {
		c.version = version
		return nil
	}
}

// WithEmbeddingsContextSize sets the context size
func WithEmbeddingsContextSize(size int) EmbeddingsOptionFunc {
	return func(c *EmbeddingsClient) error {
		c.contextSize = size
		return nil
	}
}

// WithEmbeddingsBatchSize sets the batch size
func WithEmbeddingsBatchSize(size int) EmbeddingsOptionFunc {
	return func(c *EmbeddingsClient) error {
		c.batchSize = size
		return nil
	}
}

// WithEmbeddingsPoolingType sets the pooling type
func WithEmbeddingsPoolingType(poolingType llama.PoolingType) EmbeddingsOptionFunc {
	return func(c *EmbeddingsClient) error {
		c.poolingType = poolingType
		return nil
	}
}

// WithEmbeddingsNormalize enables or disables embedding normalization
func WithEmbeddingsNormalize(normalize bool) EmbeddingsOptionFunc {
	return func(c *EmbeddingsClient) error {
		c.normalize = normalize
		return nil
	}
}

// WithEmbeddingsVerbose enables verbose logging
func WithEmbeddingsVerbose(verbose bool) EmbeddingsOptionFunc {
	return func(c *EmbeddingsClient) error {
		c.verbose = verbose
		return nil
	}
}

// NewEmbeddingsClient creates a new EmbeddingsClient
func NewEmbeddingsClient(funcs ...EmbeddingsOptionFunc) (*EmbeddingsClient, error) {
	client := &EmbeddingsClient{
		contextSize: 2048,
		batchSize:   512,
		poolingType: llama.PoolingTypeMean,
		normalize:   true,
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
func (c *EmbeddingsClient) ensureModel() error {
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
func (c *EmbeddingsClient) ensureBinaries() error {
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

// BaseEmbeddingsResponse implements llm.EmbeddingsResponse
type BaseEmbeddingsResponse struct {
	embeddings [][]float64
	usage      llm.EmbeddingsUsage
}

// Embeddings implements llm.EmbeddingsResponse
func (r *BaseEmbeddingsResponse) Embeddings() [][]float64 {
	return r.embeddings
}

// Usage implements llm.EmbeddingsResponse
func (r *BaseEmbeddingsResponse) Usage() llm.EmbeddingsUsage {
	return r.usage
}

var _ llm.EmbeddingsClient = &EmbeddingsClient{}
var _ llm.EmbeddingsResponse = &BaseEmbeddingsResponse{}

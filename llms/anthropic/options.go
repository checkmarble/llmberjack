package anthropic

import (
	"context"
)

type Backend string

const (
	BackendAnthropic Backend = "anthropic"
	BackendVertexAI  Backend = "vertexai"

	// DefaultMaxTokens is the output cap applied to requests that do not set MaxTokens.
	//
	// The Anthropic API requires max_tokens, and it counts thinking tokens, so a low
	// value truncates answers when adaptive thinking is on. It is kept as high as the
	// non-streaming path allows: this provider calls Messages.New without streaming,
	// and the SDK (CalculateNonStreamingTimeout) refuses non-streaming requests whose
	// expected duration, 1h * max_tokens / 128000, exceeds 10 minutes, that is above
	// 21333 tokens. Callers needing more must set MaxTokens explicitly, which will
	// require streaming support.
	DefaultMaxTokens = 20000

	// Deprecated: use DefaultMaxTokens, which now applies to every backend.
	BackendVertexAiDefaultMaxTokens = DefaultMaxTokens
)

type Opt func(*Anthropic)

// WithBackend sets which backend to use (anthropic or vertexai).
// Defaults to BackendAnthropic.
func WithBackend(backend Backend) Opt {
	return func(p *Anthropic) {
		p.backend = backend
	}
}

// WithApiKey sets the API key for Anthropic.
// Required for BackendAnthropic.
func WithApiKey(apiKey string) Opt {
	return func(p *Anthropic) {
		p.apiKey = apiKey
	}
}

// WithBaseUrl sets the URL at which the Anthropic API is available.
// If not specified, will use Anthropic's default API endpoint.
// Only applicable for BackendAnthropic.
func WithBaseUrl(url string) Opt {
	return func(p *Anthropic) {
		p.baseUrl = url
	}
}

// WithDefaultModel sets the default model to use if none is specified in the request.
func WithDefaultModel(model string) Opt {
	return func(p *Anthropic) {
		p.model = &model
	}
}

// WithProject sets the Google Cloud Platform project ID for Vertex AI.
// Required for BackendVertexAI.
func WithProject(project string) Opt {
	return func(p *Anthropic) {
		p.project = project
	}
}

// WithRegion sets the Google Cloud Platform region for Vertex AI.
// Defaults to "us-central1" for Vertex AI.
func WithRegion(region string) Opt {
	return func(p *Anthropic) {
		p.region = region
	}
}

// WithVertexContext sets the context for Vertex AI authentication.
// Defaults to context.Background() if not specified.
func WithVertexContext(ctx context.Context) Opt {
	return func(p *Anthropic) {
		p.vertexCtx = ctx
	}
}

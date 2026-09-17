package anthropic

import (
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/stretchr/testify/assert"
)

func TestRequestAdapter(t *testing.T) {
	llm, _ := llmberjack.New()
	p, _ := New()

	t.Run("with model", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithModel("claude-3-5-sonnet-20241022")

		params, err := p.adaptRequest(llm, req, "claude-3-5-sonnet-20241022", RequestOptions{})

		assert.Nil(t, err)
		assert.NotNil(t, params)
		assert.NotNil(t, params.Messages)
	})

	t.Run("with user prompts", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithModel("claude-3-5-sonnet-20241022").
			WithText(llmberjack.RoleUser, "user prompt", "user prompt 2").
			WithTextReader(llmberjack.RoleUser, strings.NewReader("user prompt 3"))

		params, err := p.adaptRequest(llm, req, "claude-3-5-sonnet-20241022", RequestOptions{})

		assert.Nil(t, err)
		assert.Greater(t, len(params.Messages), 0)
	})

	t.Run("with thinking level", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithModel("claude-opus-4-6").
			WithThinkingLevel(llmberjack.ThinkingLevelHigh)

		params, err := p.adaptRequest(llm, req, "claude-opus-4-6", RequestOptions{})

		assert.Nil(t, err)
		assert.NotNil(t, params.Thinking.OfAdaptive)
		assert.Equal(t, "high", string(params.OutputConfig.Effort))
	})

	t.Run("with prompt caching", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithModel("claude-opus-4-6").
			WithPromptCaching(true)

		params, err := p.adaptRequest(llm, req, "claude-opus-4-6", RequestOptions{})

		assert.Nil(t, err)
		assert.Equal(t, "ephemeral", string(params.CacheControl.Type))
	})
}

func TestBackendConfiguration(t *testing.T) {
	t.Run("anthropic backend", func(t *testing.T) {
		p, err := New(
			WithBackend(BackendAnthropic),
			WithApiKey("test-key"),
		)
		assert.Nil(t, err)
		assert.Equal(t, BackendAnthropic, p.backend)
	})

	t.Run("vertexai backend", func(t *testing.T) {
		p, err := New(
			WithBackend(BackendVertexAI),
			WithProject("test-project"),
		)
		assert.Nil(t, err)
		assert.Equal(t, BackendVertexAI, p.backend)
	})
}

func TestResponseUsage(t *testing.T) {
	llm, _ := llmberjack.New()
	p, _ := New()
	response := &anthropic.Message{
		Usage: anthropic.Usage{
			InputTokens:              20,
			OutputTokens:             30,
			CacheCreationInputTokens: 40,
			CacheReadInputTokens:     50,
		},
	}

	resp, err := p.adaptResponse(llm, response, llmberjack.NewUntypedRequest())

	assert.NoError(t, err)
	assert.Equal(t, llmberjack.ResponseUsage{
		InputTokens:      110,
		OutputTokens:     30,
		CacheWriteTokens: 40,
		CacheReadTokens:  50,
	}, resp.Usage)
}

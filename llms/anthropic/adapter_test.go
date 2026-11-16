package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/stretchr/testify/assert"
	"github.com/tidwall/gjson"
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

	t.Run("with tools", func(t *testing.T) {
		type Period struct {
			From string `json:"from" jsonschema_description:"Start date"`
			To   string `json:"to,omitempty" jsonschema_description:"End date"`
		}
		type Args struct {
			AccountId string `json:"account_id" jsonschema_description:"Account identifier"`
			Limit     int    `json:"limit,omitempty" jsonschema_description:"Maximum number of items"`
			Period    Period `json:"period" jsonschema_description:"Period to look at"`
		}

		req := llmberjack.NewUntypedRequest().
			WithModel("claude-opus-4-6").
			WithTools(llmberjack.NewTool[Args]("get_transactions", "Get transactions", llmberjack.Function(func(Args) (string, error) {
				return "", nil
			})))

		params, err := p.adaptRequest(llm, req, "claude-opus-4-6", RequestOptions{})

		assert.Nil(t, err)
		assert.Len(t, params.Tools, 1)

		body, err := json.Marshal(params.Tools[0])

		assert.Nil(t, err)

		schema := gjson.GetBytes(body, "input_schema")

		assert.Equal(t, "get_transactions", gjson.GetBytes(body, "name").String())
		assert.Equal(t, "Get transactions", gjson.GetBytes(body, "description").String())
		assert.Equal(t, "object", schema.Get("type").String())
		assert.False(t, schema.Get(`\$schema`).Exists())
		assert.Equal(t, `["account_id","limit","period"]`, schema.Get("properties|@keys").Raw)
		assert.Equal(t, "string", schema.Get("properties.account_id.type").String())
		assert.Equal(t, "Account identifier", schema.Get("properties.account_id.description").String())
		assert.Equal(t, "integer", schema.Get("properties.limit.type").String())
		assert.Equal(t, `["account_id","period"]`, schema.Get("required").Raw)
		assert.Equal(t, "false", schema.Get("additionalProperties").Raw)

		// Nested types must stay resolvable, whether inlined or referenced.
		if ref := schema.Get("properties.period.\\$ref"); ref.Exists() {
			assert.True(t, schema.Get("\\$defs.Period").Exists() || schema.Get("definitions.Period").Exists(), ref.String())
		} else {
			assert.Equal(t, `["from","to"]`, schema.Get("properties.period.properties|@keys").Raw)
		}

		// Property names must all be valid for Anthropic.
		for _, key := range schema.Get("properties|@keys").Array() {
			assert.Regexp(t, `^[a-zA-Z0-9_.-]{1,64}$`, key.String())
		}
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

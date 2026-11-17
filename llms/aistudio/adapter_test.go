package aistudio

import (
	"net/http"
	"strings"
	"testing"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/h2non/gock"
	"github.com/invopop/jsonschema"
	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"google.golang.org/genai"
)

const aistudioResponse = `{
	"responseId": "theid",
	"modelVersion": "themodel",
	"candidates": [
		{
			"finishReason": "STOP",
				"content": {
				"role": "model",
				"parts": [
					{ "text": "{\"reply\":\"The JSON response from the provider.\"}" }
				]
			}
		}
	],
	"createTime": "2025-07-13T16:20:00Z"
}`

func TestRequestAdapter(t *testing.T) {
	llm, _ := llmberjack.New()
	p, _ := New()

	t.Run("with model", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithModel("themodel")

		contents, cfg, _, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

		assert.Nil(t, err)
		assert.Nil(t, cfg.SystemInstruction)
		assert.Len(t, contents, 0)
	})

	t.Run("with system prompts", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithModel("themodel").
			WithInstruction("system prompt", "system prompt 2").
			WithInstructionReader(strings.NewReader("system prompt 3"))

		contents, cfg, _, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

		assert.Nil(t, err)
		assert.NotNil(t, cfg.SystemInstruction)
		assert.Len(t, cfg.SystemInstruction.Parts, 3)
		assert.Equal(t, "system prompt", cfg.SystemInstruction.Parts[0].Text)
		assert.Equal(t, "system prompt 2", cfg.SystemInstruction.Parts[1].Text)
		assert.Equal(t, "system prompt 3", cfg.SystemInstruction.Parts[2].Text)
		assert.Len(t, contents, 0)
	})

	t.Run("with user prompts", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithText(llmberjack.RoleUser, "user prompt", "user prompt 2").
			WithTextReader(llmberjack.RoleUser, strings.NewReader("user prompt 3"))

		contents, _, _, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

		assert.Nil(t, err)
		assert.Len(t, contents, 2)
		assert.Len(t, contents[0].Parts, 2)
		assert.Equal(t, "user prompt", contents[0].Parts[0].Text)
		assert.Equal(t, "user prompt 2", contents[0].Parts[1].Text)
		assert.Len(t, contents[1].Parts, 1)
		assert.Equal(t, "user prompt 3", contents[1].Parts[0].Text)
	})

	t.Run("with tools", func(t *testing.T) {
		type Args1 struct {
			Number int `json:"number" jsonschema_description:"Number description"`
		}
		type Args2 struct {
			Text string `json:"text" jsonschema_description:"Text description"`
		}

		req := llmberjack.NewUntypedRequest().
			WithTools(
				llmberjack.NewTool[Args1]("toolname", "tooldesc", llmberjack.Function(func(a Args1) (string, error) {
					return "", nil
				})),
				llmberjack.NewTool[Args2]("toolname 2", "tooldesc 2", llmberjack.Function(func(a Args2) (string, error) {
					return "", nil
				})),
			)

		_, cfg, _, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

		assert.Nil(t, err)
		assert.Len(t, cfg.Tools, 2)

		matchedTools := 0

		for _, tool := range cfg.Tools {
			matchedTools += 1

			assert.Len(t, tool.FunctionDeclarations, 1)

			schema, ok := tool.FunctionDeclarations[0].ParametersJsonSchema.(jsonschema.Schema)

			assert.True(t, ok)
			assert.Equal(t, "object", schema.Type)

			if tool.FunctionDeclarations[0].Name == "toolname" {
				assert.Equal(t, "tooldesc", tool.FunctionDeclarations[0].Description)
				assert.ElementsMatch(t, schema.Required, []string{"number"})
				assert.Equal(t, 1, schema.Properties.Len())
				assert.Equal(t, "integer", schema.Properties.Value("number").Type)
				assert.Equal(t, "Number description", schema.Properties.Value("number").Description)
			}
			if tool.FunctionDeclarations[0].Name == "toolname 2" {
				assert.Equal(t, "tooldesc 2", tool.FunctionDeclarations[0].Description)
				assert.ElementsMatch(t, schema.Required, []string{"text"})
				assert.Equal(t, 1, schema.Properties.Len())
				assert.Equal(t, "string", schema.Properties.Value("text").Type)
				assert.Equal(t, "Text description", schema.Properties.Value("text").Description)
			}
		}

		assert.Equal(t, 2, matchedTools)
	})

	t.Run("with response format", func(t *testing.T) {
		type Format struct {
			Text   string `json:"text" jsonschema_description:"Text description"`
			Number int    `json:"number" jsonschema_description:"Number description"`
		}

		req := llmberjack.NewRequest[Format]()

		_, cfg, _, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

		assert.Nil(t, err)
		assert.NotNil(t, cfg.ResponseJsonSchema)

		schema, ok := cfg.ResponseJsonSchema.(*jsonschema.Schema)

		assert.True(t, ok)
		assert.Equal(t, "object", schema.Type)
		assert.ElementsMatch(t, schema.Required, []string{"text", "number"})
		assert.Equal(t, 2, schema.Properties.Len())
		assert.Equal(t, "string", schema.Properties.Value("text").Type)
		assert.Equal(t, "Text description", schema.Properties.Value("text").Description)
		assert.Equal(t, "integer", schema.Properties.Value("number").Type)
		assert.Equal(t, "Number description", schema.Properties.Value("number").Description)
	})

	t.Run("with request options", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithMaxCandidates(10).
			WithMaxTokens(42).
			WithTemperature(0.1).
			WithTopP(0.1)

		_, cfg, _, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

		assert.Nil(t, err)
		assert.EqualValues(t, 10, cfg.CandidateCount)
		assert.EqualValues(t, 42, cfg.MaxOutputTokens)
		assert.Equal(t, float32(0.1), *cfg.Temperature)
		assert.Equal(t, float32(0.1), *cfg.TopP)
	})

	t.Run("with provider options", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest()
		opts := RequestOptions{
			GoogleSearch: lo.ToPtr(true),
			TopK:         lo.ToPtr(0.2),
		}

		_, cfg, _, err := p.adaptRequest(llm, req, opts)

		assert.Nil(t, err)

		matchedTools := 0

		for _, tool := range cfg.Tools {
			matchedTools += 1

			assert.NotNil(t, tool.GoogleSearch)
		}

		assert.Equal(t, 1, matchedTools)
		assert.NotNil(t, cfg.TopK)
		assert.Equal(t, float32(0.2), *cfg.TopK)
	})
}

func TestSkipHistory(t *testing.T) {
	defer gock.Off()

	httpClient := &http.Client{}
	provider, _ := New(WithBackend(genai.BackendVertexAI), WithLocation("location"), WithProject("project"))
	llm, _ := llmberjack.New(llmberjack.WithDefaultProvider(provider), llmberjack.WithDefaultModel("themodel"), llmberjack.WithHttpClient(httpClient))

	gock.InterceptClient(httpClient)

	gock.New("https://location-aiplatform.googleapis.com").
		Post("/v1beta1/projects/project/locations/location/publishers/google/models/themodel:generateContent").
		Persist().
		Reply(http.StatusOK).
		SetHeader("content-type", "application/json").
		BodyString(aistudioResponse)

	resp, _ := llmberjack.NewUntypedRequest().CreateThread().WithInstruction("system text").Do(t.Context(), llm)
	threadId := resp.ThreadId

	assert.Len(t, provider.history.Load(threadId), 1)

	_, _ = llmberjack.NewUntypedRequest().InThread(threadId).WithInstruction("system text").Do(t.Context(), llm)

	assert.Len(t, provider.history.Load(threadId), 2)

	resp, _ = llmberjack.NewUntypedRequest().InThread(threadId).WithInstruction("system text").SkipSaveInput().Do(t.Context(), llm)

	assert.Len(t, provider.history.Load(resp.ThreadId), 2)

	resp.Candidates[0].SelectCandidate()

	assert.Len(t, provider.history.Load(threadId), 3)
}

package openai

import (
	"strings"
	"testing"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/invopop/jsonschema"
	"github.com/stretchr/testify/assert"
)

func TestRequestAdapter(t *testing.T) {
	llm, _ := llmberjack.New()
	p, _ := New()

	t.Run("with model", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithModel("themodel")

		cfg, _, err := p.adaptRequest(llm, req)

		assert.Nil(t, err)
		assert.Equal(t, "themodel", cfg.Model)
	})

	t.Run("with system prompts", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithModel("themodel").
			WithInstruction("system prompt", "system prompt 2").
			WithInstructionReader(strings.NewReader("system prompt 3"))

		cfg, _, err := p.adaptRequest(llm, req)

		assert.Nil(t, err)
		assert.Len(t, cfg.Messages, 2)
		assert.Len(t, cfg.Messages[0].OfSystem.Content.OfArrayOfContentParts, 2)
		assert.Equal(t, "system prompt", cfg.Messages[0].OfSystem.Content.OfArrayOfContentParts[0].Text)
		assert.Equal(t, "system prompt 2", cfg.Messages[0].OfSystem.Content.OfArrayOfContentParts[1].Text)
		assert.Equal(t, "system prompt 3", cfg.Messages[1].OfSystem.Content.OfArrayOfContentParts[0].Text)
	})

	t.Run("with user prompts", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithText(llmberjack.RoleUser, "user prompt", "user prompt 2").
			WithTextReader(llmberjack.RoleUser, strings.NewReader("user prompt 3"))

		cfg, _, err := p.adaptRequest(llm, req)

		assert.Nil(t, err)
		assert.Len(t, cfg.Messages, 2)
		assert.Len(t, cfg.Messages[0].OfUser.Content.OfArrayOfContentParts, 2)
		assert.Equal(t, "user prompt", cfg.Messages[0].OfUser.Content.OfArrayOfContentParts[0].OfText.Text)
		assert.Equal(t, "user prompt 2", cfg.Messages[0].OfUser.Content.OfArrayOfContentParts[1].OfText.Text)
		assert.Equal(t, "user prompt 3", cfg.Messages[1].OfUser.Content.OfArrayOfContentParts[0].OfText.Text)
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

		cfg, _, err := p.adaptRequest(llm, req)

		assert.Nil(t, err)
		assert.Len(t, cfg.Tools, 2)

		matchedTools := 0

		for _, tool := range cfg.Tools {
			matchedTools += 1

			schema := tool.Function.Parameters

			assert.Equal(t, "object", schema["type"])

			if tool.Function.Name == "toolname" {
				assert.Equal(t, "tooldesc", tool.Function.Description.Value)
				assert.ElementsMatch(t, schema["required"], []string{"number"})
				assert.Len(t, schema["properties"], 1)
				assert.Equal(t, "integer", schema["properties"].(map[string]any)["number"].(map[string]any)["type"])
				assert.Equal(t, "Number description", schema["properties"].(map[string]any)["number"].(map[string]any)["description"])
			}
			if tool.Function.Name == "toolname 2" {
				assert.Equal(t, "tooldesc 2", tool.Function.Description.Value)
				assert.ElementsMatch(t, schema["required"], []string{"text"})
				assert.Len(t, schema["properties"], 1)
				assert.Equal(t, "string", schema["properties"].(map[string]any)["text"].(map[string]any)["type"])
				assert.Equal(t, "Text description", schema["properties"].(map[string]any)["text"].(map[string]any)["description"])
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

		cfg, _, err := p.adaptRequest(llm, req)

		assert.Nil(t, err)
		assert.NotNil(t, cfg.ResponseFormat.OfJSONSchema)

		schema, ok := cfg.ResponseFormat.OfJSONSchema.JSONSchema.Schema.(*jsonschema.Schema)

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

		cfg, _, err := p.adaptRequest(llm, req)

		assert.Nil(t, err)
		assert.EqualValues(t, 10, cfg.N.Value)
		assert.EqualValues(t, 42, cfg.MaxTokens.Value)
		assert.Equal(t, 0.1, cfg.Temperature.Value)
		assert.Equal(t, 0.1, cfg.TopP.Value)
	})
}

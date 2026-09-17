package openai_test

import (
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/checkmarble/llmberjack/llms/openai"
	"github.com/h2non/gock"
	"github.com/stretchr/testify/assert"
	"github.com/tidwall/gjson"
)

const openaiResponse = `{
	"id": "theid",
	"model": "themodel",
	"choices": [
		{
			"index": 0,
			"finish_reason": "stop",
			"message": {
				"type": "message",
				"role": "assistant",
				"content": "{\"reply\":\"The JSON response from the provider.\"}"
			}
		}
	],
	"created": 1752423600,
	"usage": {
		"prompt_tokens": 120,
		"completion_tokens": 30,
		"total_tokens": 150,
		"prompt_tokens_details": {
			"cached_tokens": 80
		}
	}
}`

func TestOpenAiRequest(t *testing.T) {
	defer gock.Off()

	type Output struct {
		Reply string `json:"reply" jsonschema_description:"Write your response here"`
	}

	type Args struct {
		Name string `json:"name" jsonschema_description:"My name"`
	}

	provider, _ := openai.New(openai.WithApiKey("apikey"))
	llm, _ := llmberjack.New(llmberjack.WithDefaultProvider(provider))

	req := llmberjack.NewRequest[Output]().
		WithModel("themodel").
		WithThinkingLevel(llmberjack.ThinkingLevelLow).
		WithPromptCaching(true).
		WithInstruction("system text").
		WithInstructionReader(strings.NewReader("text from reader")).
		WithText(llmberjack.RoleUser, "user text").
		WithTools(llmberjack.NewTool[Args]("thetool", "Tool to get nothing", llmberjack.Function(func(Args) (string, error) {
			return "OK", nil
		}))).
		WithTextReader(llmberjack.RoleUser, strings.NewReader("text from reader"))

	gock.New("https://api.openai.com").
		Post("/v1/chat/completions").
		MatchHeader("authorization", "Bearer apikey").
		AddMatcher(func(req *http.Request, _ *gock.Request) (bool, error) {
			body, _ := io.ReadAll(req.Body)

			assert.Equal(t, "themodel", gjson.GetBytes(body, "model").String())
			assert.Equal(t, "low", gjson.GetBytes(body, "reasoning_effort").String())
			assert.False(t, gjson.GetBytes(body, "cache_control").Exists())

			assert.EqualValues(t, 4, gjson.GetBytes(body, "messages.#").Int())
			assert.Equal(t, "system text", gjson.GetBytes(body, "messages.0.content.0.text").String())
			assert.Equal(t, "system", gjson.GetBytes(body, "messages.0.role").String())
			assert.Equal(t, "text from reader", gjson.GetBytes(body, "messages.1.content.0.text").String())
			assert.Equal(t, "system", gjson.GetBytes(body, "messages.1.role").String())
			assert.Equal(t, "user text", gjson.GetBytes(body, "messages.2.content.0.text").String())
			assert.Equal(t, "user", gjson.GetBytes(body, "messages.2.role").String())
			assert.Equal(t, "text from reader", gjson.GetBytes(body, "messages.3.content.0.text").String())
			assert.Equal(t, "user", gjson.GetBytes(body, "messages.3.role").String())

			assert.Equal(t, "object", gjson.GetBytes(body, "response_format.json_schema.schema.type").String())
			assert.EqualValues(t, 1, gjson.GetBytes(body, "response_format.json_schema.schema.properties|@keys|#").Int())
			assert.EqualValues(t, 1, gjson.GetBytes(body, "response_format.json_schema.schema.required.#").Int())
			assert.Equal(t, "string", gjson.GetBytes(body, "response_format.json_schema.schema.properties.reply.type").String())
			assert.Equal(t, "Write your response here", gjson.GetBytes(body, "response_format.json_schema.schema.properties.reply.description").String())

			assert.EqualValues(t, 1, gjson.GetBytes(body, "tools.#").Int())
			assert.Equal(t, "thetool", gjson.GetBytes(body, "tools.0.function.name").String())
			assert.Equal(t, "Tool to get nothing", gjson.GetBytes(body, "tools.0.function.description").String())
			assert.EqualValues(t, 1, gjson.GetBytes(body, "tools.0.function.parameters.properties|@keys|#").Int())
			assert.Equal(t, "string", gjson.GetBytes(body, "tools.0.function.parameters.properties.name.type").String())
			assert.Equal(t, "My name", gjson.GetBytes(body, "tools.0.function.parameters.properties.name.description").String())

			return true, nil
		}).
		Reply(http.StatusOK).
		SetHeader("content-type", "application/json").BodyString(openaiResponse)

	resp, err := req.Do(t.Context(), llm)

	assert.False(t, gock.HasUnmatchedRequest())
	assert.Nil(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "themodel", resp.Model)
	assert.Equal(t, "theid", resp.Id)
	assert.WithinDuration(t, time.Date(2025, 7, 13, 16, 20, 0, 0, time.UTC), resp.Created, 0)
	assert.Equal(t, llmberjack.ResponseUsage{
		InputTokens:     120,
		OutputTokens:    30,
		CacheReadTokens: 80,
	}, resp.Usage)
	assert.Equal(t, 1, resp.NumCandidates())

	candidate, err := resp.Candidate(0)

	assert.Nil(t, err)
	assert.Equal(t, llmberjack.FinishReasonStop, candidate.FinishReason)

	output, err := resp.Get(0)

	assert.Nil(t, err)
	assert.Equal(t, "The JSON response from the provider.", output.Reply)
}

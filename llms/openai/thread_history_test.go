package openai

import (
	"io"
	"net/http"
	"testing"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/h2non/gock"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/tidwall/gjson"
)

const (
	threadHistoryUrl = "https://api.openai.com"

	toolCallResponse = `{
		"id": "theid",
		"model": "themodel",
		"created": 1752423600,
		"choices": [
			{
				"index": 0,
				"finish_reason": "tool_calls",
				"message": {
					"role": "assistant",
					"content": null,
					"tool_calls": [
						{ "id": "call1", "type": "function", "function": { "name": "thetool", "arguments": "{\"name\":\"value\"}" } }
					]
				}
			}
		]
	}`
	textResponse = `{
		"id": "theid",
		"model": "themodel",
		"created": 1752423600,
		"choices": [
			{ "index": 0, "finish_reason": "stop", "message": { "role": "assistant", "content": "the answer" } }
		]
	}`
	errorResponse = `{ "error": { "message": "bad request", "type": "invalid_request_error" } }`
)

func newThreadHistoryProvider(t *testing.T) (*OpenAi, *llmberjack.Llmberjack) {
	t.Helper()

	httpClient := &http.Client{}
	provider, _ := New(WithApiKey("apikey"))
	llm, err := llmberjack.New(
		llmberjack.WithDefaultProvider(provider),
		llmberjack.WithDefaultModel("themodel"),
		llmberjack.WithHttpClient(httpClient),
	)
	assert.NoError(t, err)

	gock.InterceptClient(httpClient)

	return provider, llm
}

func mockChatCompletion(status int, body string, captured *[]byte) {
	m := gock.New(threadHistoryUrl).Post("/v1/chat/completions")

	if captured != nil {
		m = m.AddMatcher(func(req *http.Request, _ *gock.Request) (bool, error) {
			*captured, _ = io.ReadAll(req.Body)
			return true, nil
		})
	}

	m.Reply(status).SetHeader("content-type", "application/json").BodyString(body)
}

func TestFailedFirstTurnLeavesNoHistory(t *testing.T) {
	defer gock.Off()

	provider, llm := newThreadHistoryProvider(t)
	threadId := &llmberjack.ThreadId{}

	// Register another thread so that loading an unknown thread returns nil,
	// telling it apart from a registered but empty one.
	provider.history.Save(&llmberjack.ThreadId{}, openai.UserMessage("other"))

	mockChatCompletion(http.StatusBadRequest, errorResponse, nil)

	req := llmberjack.NewUntypedRequest().
		InThread(threadId).
		WithInstruction("system text").
		WithText(llmberjack.RoleUser, "user text")

	_, err := provider.ChatCompletion(t.Context(), llm, req)

	assert.Error(t, err)
	assert.Nil(t, provider.history.Load(threadId), "a failed first turn must not register the thread")
	assert.True(t, gock.IsDone())
}

func TestFailedTurnRetryInThread(t *testing.T) {
	defer gock.Off()

	type Args struct {
		Name string `json:"name"`
	}

	provider, llm := newThreadHistoryProvider(t)
	tool := llmberjack.NewTool[Args]("thetool", "A tool", llmberjack.Function(func(Args) (string, error) {
		return "tool output", nil
	}))

	var retryBody []byte

	mockChatCompletion(http.StatusOK, toolCallResponse, nil)
	mockChatCompletion(http.StatusBadRequest, errorResponse, nil)
	mockChatCompletion(http.StatusOK, textResponse, &retryBody)

	// First turn succeeds: the instruction and the prompt are committed.
	resp, err := llmberjack.NewUntypedRequest().
		CreateThread().
		WithInstruction("system text").
		WithText(llmberjack.RoleUser, "user text").
		WithTools(tool).
		Do(t.Context(), llm)

	assert.NoError(t, err)
	assert.Len(t, provider.history.Load(resp.ThreadId), 2)

	// Second turn (tool response) fails: the history is left as it was
	// before the call (instruction, prompt and the selected tool call).
	_, err = llmberjack.NewUntypedRequest().
		FromCandidate(resp, 0).
		WithToolExecution(tool).
		Do(t.Context(), llm)

	assert.Error(t, err)

	history := provider.history.Load(resp.ThreadId)

	assert.Len(t, history, 3)
	assert.Equal(t, "call1", history[2].OfAssistant.ToolCalls[0].ID)

	// Retrying the very same turn sends each message only once, and commits
	// the tool response on success.
	retry, err := llmberjack.NewUntypedRequest().
		FromCandidate(resp, 0).
		WithToolExecution(tool).
		Do(t.Context(), llm)

	assert.NoError(t, err)
	assert.Equal(t, resp.ThreadId, retry.ThreadId)

	assert.EqualValues(t, 4, gjson.GetBytes(retryBody, "messages.#").Int())
	assert.Equal(t, "system", gjson.GetBytes(retryBody, "messages.0.role").String())
	assert.Equal(t, "user", gjson.GetBytes(retryBody, "messages.1.role").String())
	assert.Equal(t, "call1", gjson.GetBytes(retryBody, "messages.2.tool_calls.0.id").String())
	assert.Equal(t, "call1", gjson.GetBytes(retryBody, "messages.3.tool_call_id").String())

	history = provider.history.Load(resp.ThreadId)

	assert.Len(t, history, 4)
	assert.Equal(t, "call1", history[3].OfTool.ToolCallID)

	// Selecting the answer appends it as before.
	retry.Candidates[0].SelectCandidate()

	assert.Len(t, provider.history.Load(resp.ThreadId), 5)
	assert.True(t, gock.IsDone())
}

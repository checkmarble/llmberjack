package aistudio

import (
	"io"
	"net/http"
	"testing"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/h2non/gock"
	"github.com/stretchr/testify/assert"
	"github.com/tidwall/gjson"
	"google.golang.org/genai"
)

const (
	threadHistoryUrl  = "https://location-aiplatform.googleapis.com"
	threadHistoryPath = "/v1beta1/projects/project/locations/location/publishers/google/models/themodel:generateContent"

	toolCallResponse = `{
		"responseId": "theid",
		"modelVersion": "themodel",
		"candidates": [
			{
				"finishReason": "STOP",
				"content": {
					"role": "model",
					"parts": [
						{ "functionCall": { "id": "call1", "name": "thetool", "args": { "name": "value" } } }
					]
				}
			}
		]
	}`
	errorResponse = `{ "error": { "code": 400, "message": "bad request", "status": "INVALID_ARGUMENT" } }`
)

func newThreadHistoryProvider(t *testing.T) (*AiStudio, *llmberjack.Llmberjack) {
	t.Helper()

	httpClient := &http.Client{}
	provider, _ := New(WithBackend(genai.BackendVertexAI), WithLocation("location"), WithProject("project"))
	llm, err := llmberjack.New(
		llmberjack.WithDefaultProvider(provider),
		llmberjack.WithDefaultModel("themodel"),
		llmberjack.WithHttpClient(httpClient),
	)
	assert.NoError(t, err)

	gock.InterceptClient(httpClient)

	return provider, llm
}

func mockGenerateContent(status int, body string, captured *[]byte) {
	m := gock.New(threadHistoryUrl).Post(threadHistoryPath)

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
	provider.history.Save(&llmberjack.ThreadId{}, &genai.Content{})

	mockGenerateContent(http.StatusBadRequest, errorResponse, nil)

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

	mockGenerateContent(http.StatusOK, toolCallResponse, nil)
	mockGenerateContent(http.StatusBadRequest, errorResponse, nil)
	mockGenerateContent(http.StatusOK, aistudioResponse, &retryBody)

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
	assert.Equal(t, "call1", history[2].Parts[0].FunctionCall.ID)

	// Retrying the very same turn sends each message only once, and commits
	// the tool response on success.
	retry, err := llmberjack.NewUntypedRequest().
		FromCandidate(resp, 0).
		WithToolExecution(tool).
		Do(t.Context(), llm)

	assert.NoError(t, err)
	assert.Equal(t, resp.ThreadId, retry.ThreadId)

	assert.EqualValues(t, 4, gjson.GetBytes(retryBody, "contents.#").Int())
	assert.Equal(t, "system text", gjson.GetBytes(retryBody, "contents.0.parts.0.text").String())
	assert.Equal(t, "user text", gjson.GetBytes(retryBody, "contents.1.parts.0.text").String())
	assert.Equal(t, "thetool", gjson.GetBytes(retryBody, "contents.2.parts.0.functionCall.name").String())
	assert.Equal(t, "thetool", gjson.GetBytes(retryBody, "contents.3.parts.0.functionResponse.name").String())

	history = provider.history.Load(resp.ThreadId)

	assert.Len(t, history, 4)
	assert.Equal(t, "thetool", history[3].Parts[0].FunctionResponse.Name)

	// Selecting the answer appends it as before.
	retry.Candidates[0].SelectCandidate()

	assert.Len(t, provider.history.Load(resp.ThreadId), 5)
	assert.True(t, gock.IsDone())
}

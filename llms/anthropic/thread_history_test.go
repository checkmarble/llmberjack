package anthropic

import (
	"io"
	"net/http"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/h2non/gock"
	"github.com/stretchr/testify/assert"
	"github.com/tidwall/gjson"
)

const (
	threadHistoryUrl = "https://api.anthropic.com"

	textResponse = `{
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "themodel",
		"content": [ { "type": "text", "text": "the answer" } ],
		"stop_reason": "end_turn",
		"usage": { "input_tokens": 10, "output_tokens": 5 }
	}`
	errorResponse = `{ "type": "error", "error": { "type": "invalid_request_error", "message": "bad request" } }`
)

func newThreadHistoryProvider(t *testing.T) (*Anthropic, *llmberjack.Llmberjack) {
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

func mockMessages(status int, body string, captured *[]byte) {
	m := gock.New(threadHistoryUrl).Post("/v1/messages")

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
	provider.history.Save(&llmberjack.ThreadId{}, anthropic.NewUserMessage(anthropic.NewTextBlock("other")))

	mockMessages(http.StatusBadRequest, errorResponse, nil)

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

	provider, llm := newThreadHistoryProvider(t)

	var retryBody []byte

	mockMessages(http.StatusOK, textResponse, nil)
	mockMessages(http.StatusBadRequest, errorResponse, nil)
	mockMessages(http.StatusOK, textResponse, &retryBody)

	// First turn succeeds: the prompt is committed.
	resp, err := llmberjack.NewUntypedRequest().
		CreateThread().
		WithInstruction("system text").
		WithText(llmberjack.RoleUser, "first prompt").
		Do(t.Context(), llm)

	assert.NoError(t, err)
	assert.Len(t, provider.history.Load(resp.ThreadId), 1)

	// Second turn fails: the history is left as it was before the call (the
	// first prompt and the selected answer).
	_, err = llmberjack.NewUntypedRequest().
		FromCandidate(resp, 0).
		WithText(llmberjack.RoleUser, "second prompt").
		Do(t.Context(), llm)

	assert.Error(t, err)
	assert.Len(t, provider.history.Load(resp.ThreadId), 2)

	// Retrying the very same turn sends each message only once, and commits
	// the new prompt on success.
	retry, err := llmberjack.NewUntypedRequest().
		FromCandidate(resp, 0).
		WithText(llmberjack.RoleUser, "second prompt").
		Do(t.Context(), llm)

	assert.NoError(t, err)
	assert.Equal(t, resp.ThreadId, retry.ThreadId)

	assert.EqualValues(t, 3, gjson.GetBytes(retryBody, "messages.#").Int())
	assert.Equal(t, "first prompt", gjson.GetBytes(retryBody, "messages.0.content.0.text").String())
	assert.Equal(t, "the answer", gjson.GetBytes(retryBody, "messages.1.content.0.text").String())
	assert.Equal(t, "second prompt", gjson.GetBytes(retryBody, "messages.2.content.0.text").String())

	assert.Len(t, provider.history.Load(resp.ThreadId), 3)

	// Selecting the answer appends it as before.
	retry.Candidates[0].SelectCandidate()

	assert.Len(t, provider.history.Load(resp.ThreadId), 4)
	assert.True(t, gock.IsDone())
}

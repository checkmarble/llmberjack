package anthropic

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/checkmarble/llmberjack/internal"
	"github.com/h2non/gock"
	"github.com/stretchr/testify/assert"
	"github.com/tidwall/gjson"
)

type toolArgs struct {
	Value string `json:"value"`
}

type fakeCandidater struct {
	thread    *llmberjack.ThreadId
	candidate llmberjack.ResponseCandidate
}

func (f fakeCandidater) NumCandidates() int { return 1 }
func (f fakeCandidater) Candidate(int) (*llmberjack.ResponseCandidate, error) {
	return &f.candidate, nil
}
func (f fakeCandidater) Thread() *llmberjack.ThreadId { return f.thread }

func echoTool(name string) internal.Tool {
	return llmberjack.NewTool[toolArgs](name, name, llmberjack.Function(func(a toolArgs) (string, error) {
		return name + ":" + a.Value, nil
	}))
}

func toolResultIds(msg anthropic.MessageParam) []string {
	ids := make([]string, 0, len(msg.Content))

	for _, block := range msg.Content {
		if block.OfToolResult != nil {
			ids = append(ids, block.OfToolResult.ToolUseID)
		}
	}

	return ids
}

func TestParallelToolResultsAreGrouped(t *testing.T) {
	llm, _ := llmberjack.New()
	p, _ := New()

	threadId := &llmberjack.ThreadId{}
	previous := fakeCandidater{
		thread: threadId,
		candidate: llmberjack.ResponseCandidate{
			ToolCalls: []llmberjack.ResponseToolCall{
				{Id: "toolu_1", Name: "tool_a", Parameters: []byte(`{"value":"1"}`)},
				{Id: "toolu_2", Name: "tool_b", Parameters: []byte(`{"value":"2"}`)},
				{Id: "toolu_3", Name: "tool_a", Parameters: []byte(`{"value":"3"}`)},
			},
			SelectCandidate: func() {},
		},
	}

	req := llmberjack.NewUntypedRequest().
		FromCandidate(previous, 0).
		WithToolExecution(echoTool("tool_a"), echoTool("tool_b")).
		WithText(llmberjack.RoleAi, "intermediate assistant message").
		WithToolExecution(echoTool("tool_a"), echoTool("tool_b"))

	params, err := p.adaptRequest(llm, req, "claude-opus-4-6", RequestOptions{})

	assert.Nil(t, err)
	assert.Len(t, params.Messages, 3)

	assert.Equal(t, anthropic.MessageParamRoleUser, params.Messages[0].Role)
	assert.Equal(t, []string{"toolu_1", "toolu_2", "toolu_3"}, toolResultIds(params.Messages[0]))

	assert.Equal(t, anthropic.MessageParamRoleAssistant, params.Messages[1].Role)

	assert.Equal(t, anthropic.MessageParamRoleUser, params.Messages[2].Role)
	assert.Equal(t, []string{"toolu_1", "toolu_2", "toolu_3"}, toolResultIds(params.Messages[2]))

	// The thread history holds the same grouped messages, in the same order.
	history := p.history.Load(threadId)

	assert.Len(t, history, 3)
	assert.Equal(t, []string{"toolu_1", "toolu_2", "toolu_3"}, toolResultIds(history[0]))
	assert.Equal(t, anthropic.MessageParamRoleAssistant, history[1].Role)
	assert.Equal(t, []string{"toolu_1", "toolu_2", "toolu_3"}, toolResultIds(history[2]))
}

const anthropicToolUseResponse = `{
	"id": "msg_%[1]d",
	"type": "message",
	"role": "assistant",
	"model": "claude-opus-4-6",
	"content": [%[2]s],
	"stop_reason": "%[3]s",
	"usage": { "input_tokens": 10, "output_tokens": 10 }
}`

func toolUseResponse(id int, calls ...[3]string) string {
	blocks := []string{
		`{ "type": "thinking", "thinking": "Let me call the tools.", "signature": "sig_` + fmt.Sprint(id) + `" }`,
		`{ "type": "text", "text": "Calling tools." }`,
	}

	for _, call := range calls {
		blocks = append(blocks, fmt.Sprintf(`{ "type": "tool_use", "id": %q, "name": %q, "input": { "value": %q } }`, call[0], call[1], call[2]))
	}

	return fmt.Sprintf(anthropicToolUseResponse, id, strings.Join(blocks, ","), "tool_use")
}

func TestParallelToolCallsInThread(t *testing.T) {
	defer gock.Off()

	httpClient := &http.Client{}
	provider, _ := New(WithBackend(BackendAnthropic), WithApiKey("key"), WithBaseUrl("https://api.anthropic.test/"))
	llm, _ := llmberjack.New(llmberjack.WithDefaultProvider(provider), llmberjack.WithDefaultModel("claude-opus-4-6"), llmberjack.WithHttpClient(httpClient))

	gock.InterceptClient(httpClient)

	var turn2, turn3 []byte

	gock.New("https://api.anthropic.test").Post("/v1/messages").
		Reply(http.StatusOK).
		SetHeader("content-type", "application/json").
		BodyString(toolUseResponse(1, [3]string{"toolu_1", "tool_a", "1"}, [3]string{"toolu_2", "tool_b", "2"}, [3]string{"toolu_3", "tool_a", "3"}))

	gock.New("https://api.anthropic.test").Post("/v1/messages").
		AddMatcher(func(req *http.Request, _ *gock.Request) (bool, error) {
			turn2, _ = io.ReadAll(req.Body)
			return true, nil
		}).
		Reply(http.StatusOK).
		SetHeader("content-type", "application/json").
		BodyString(toolUseResponse(2, [3]string{"toolu_4", "tool_b", "4"}, [3]string{"toolu_5", "tool_a", "5"}))

	gock.New("https://api.anthropic.test").Post("/v1/messages").
		AddMatcher(func(req *http.Request, _ *gock.Request) (bool, error) {
			turn3, _ = io.ReadAll(req.Body)
			return true, nil
		}).
		Reply(http.StatusOK).
		SetHeader("content-type", "application/json").
		BodyString(fmt.Sprintf(anthropicToolUseResponse, 3, `{ "type": "text", "text": "Done." }`, "end_turn"))

	tools := []internal.Tool{echoTool("tool_a"), echoTool("tool_b")}

	resp1, err := llmberjack.NewUntypedRequest().CreateThread().WithTools(tools...).WithText(llmberjack.RoleUser, "start").Do(t.Context(), llm)
	assert.Nil(t, err)

	resp2, err := llmberjack.NewUntypedRequest().FromCandidate(resp1, 0).WithToolExecution(tools...).Do(t.Context(), llm)
	assert.Nil(t, err)

	resp3, err := llmberjack.NewUntypedRequest().FromCandidate(resp2, 0).WithToolExecution(tools...).Do(t.Context(), llm)
	assert.Nil(t, err)

	assert.False(t, gock.HasUnmatchedRequest())
	assert.Equal(t, "Done.", resp3.Candidates[0].Text)

	// Turn 2: user, the full assistant turn (thinking with its signature,
	// text and the 3 tool_use blocks), then ONE user message with the 3
	// tool_result blocks.
	assert.EqualValues(t, 3, gjson.GetBytes(turn2, "messages.#").Int())
	assert.Equal(t, "assistant", gjson.GetBytes(turn2, "messages.1.role").String())
	assert.Equal(t, `["thinking","text","tool_use","tool_use","tool_use"]`, gjson.GetBytes(turn2, "messages.1.content.#.type").Raw)
	assert.Equal(t, "sig_1", gjson.GetBytes(turn2, "messages.1.content.0.signature").String())
	assert.Equal(t, `["toolu_1","toolu_2","toolu_3"]`, gjson.GetBytes(turn2, "messages.1.content.#(type==tool_use)#.id").Raw)
	assert.Equal(t, "user", gjson.GetBytes(turn2, "messages.2.role").String())
	assert.Equal(t, `["tool_result","tool_result","tool_result"]`, gjson.GetBytes(turn2, "messages.2.content.#.type").Raw)
	assert.Equal(t, `["toolu_1","toolu_2","toolu_3"]`, gjson.GetBytes(turn2, "messages.2.content.#.tool_use_id").Raw)
	assert.Equal(t, `["tool_a:1","tool_b:2","tool_a:3"]`, gjson.GetBytes(turn2, "messages.2.content.#.content.0.text").Raw)

	// Turn 3: the history replays the first batch as a single message, and
	// the second batch, after the second assistant turn, is a separate one.
	assert.EqualValues(t, 5, gjson.GetBytes(turn3, "messages.#").Int())
	assert.Equal(t, `["toolu_1","toolu_2","toolu_3"]`, gjson.GetBytes(turn3, "messages.2.content.#.tool_use_id").Raw)
	assert.Equal(t, "assistant", gjson.GetBytes(turn3, "messages.3.role").String())
	assert.Equal(t, `["toolu_4","toolu_5"]`, gjson.GetBytes(turn3, "messages.3.content.#(type==tool_use)#.id").Raw)
	assert.Equal(t, "user", gjson.GetBytes(turn3, "messages.4.role").String())
	assert.Equal(t, `["toolu_4","toolu_5"]`, gjson.GetBytes(turn3, "messages.4.content.#.tool_use_id").Raw)
}

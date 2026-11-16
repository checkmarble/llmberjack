package aistudio

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/checkmarble/llmberjack/internal"
	"github.com/h2non/gock"
	"github.com/invopop/jsonschema"
	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"github.com/tidwall/gjson"
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

		contents, cfg, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

		assert.Nil(t, err)
		assert.Nil(t, cfg.SystemInstruction)
		assert.Len(t, contents, 0)
	})

	t.Run("with system prompts", func(t *testing.T) {
		req := llmberjack.NewUntypedRequest().
			WithModel("themodel").
			WithInstruction("system prompt", "system prompt 2").
			WithInstructionReader(strings.NewReader("system prompt 3"))

		contents, cfg, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

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

		contents, _, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

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

		_, cfg, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

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

		_, cfg, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

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

		_, cfg, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

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

		_, cfg, err := p.adaptRequest(llm, req, opts)

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

func assertFunctionResponses(t *testing.T, content *genai.Content, expected ...[2]string) {
	t.Helper()

	assert.Equal(t, genai.RoleUser, content.Role)
	assert.Len(t, content.Parts, len(expected))

	for idx, exp := range expected {
		if idx >= len(content.Parts) {
			return
		}

		fr := content.Parts[idx].FunctionResponse

		assert.NotNil(t, fr)
		if fr == nil {
			continue
		}

		assert.Equal(t, exp[0], fr.ID)
		assert.Equal(t, exp[1], fr.Name)
	}
}

func TestParallelToolResponsesAreGrouped(t *testing.T) {
	llm, _ := llmberjack.New()
	p, _ := New()

	threadId := &llmberjack.ThreadId{}
	previous := fakeCandidater{
		thread: threadId,
		candidate: llmberjack.ResponseCandidate{
			ToolCalls: []llmberjack.ResponseToolCall{
				{Id: "call_1", Name: "tool_a", Parameters: []byte(`{"value":"1"}`)},
				{Id: "call_2", Name: "tool_b", Parameters: []byte(`{"value":"2"}`)},
				{Id: "call_3", Name: "tool_a", Parameters: []byte(`{"value":"3"}`)},
			},
			SelectCandidate: func() {},
		},
	}

	req := llmberjack.NewUntypedRequest().
		FromCandidate(previous, 0).
		WithToolExecution(echoTool("tool_a"), echoTool("tool_b")).
		WithText(llmberjack.RoleAi, "intermediate model message").
		WithToolExecution(echoTool("tool_a"), echoTool("tool_b"))

	contents, _, err := p.adaptRequest(llm, req, lo.FromPtr[RequestOptions](nil))

	assert.Nil(t, err)
	assert.Len(t, contents, 3)

	expected := [][2]string{{"call_1", "tool_a"}, {"call_2", "tool_b"}, {"call_3", "tool_a"}}

	assertFunctionResponses(t, contents[0], expected...)
	assert.Equal(t, "tool_a:1", contents[0].Parts[0].FunctionResponse.Response["output"].(*genai.Part).Text)
	assert.Equal(t, "tool_b:2", contents[0].Parts[1].FunctionResponse.Response["output"].(*genai.Part).Text)
	assert.Equal(t, "tool_a:3", contents[0].Parts[2].FunctionResponse.Response["output"].(*genai.Part).Text)

	assert.Equal(t, genai.RoleModel, contents[1].Role)
	assert.Equal(t, "intermediate model message", contents[1].Parts[0].Text)

	assertFunctionResponses(t, contents[2], expected...)
	assert.NotSame(t, contents[0], contents[2])

	// The thread history must contain the grouped contents, in the same order,
	// so that the next turn replays a valid conversation.
	history := p.history.Load(threadId)

	assert.Len(t, history, 3)
	assert.Same(t, contents[0], history[0])
	assert.Same(t, contents[1], history[1])
	assert.Same(t, contents[2], history[2])
}

const aistudioParallelCallsResponse = `{
	"responseId": "theid",
	"modelVersion": "themodel",
	"candidates": [
		{
			"finishReason": "STOP",
			"content": {
				"role": "model",
				"parts": [%s]
			}
		}
	],
	"createTime": "2025-07-13T16:20:00Z"
}`

func functionCallsResponse(calls ...[3]string) string {
	parts := make([]string, len(calls))

	for idx, call := range calls {
		parts[idx] = fmt.Sprintf(`{ "functionCall": { "id": %q, "name": %q, "args": { "value": %q } } }`, call[0], call[1], call[2])
	}

	return fmt.Sprintf(aistudioParallelCallsResponse, strings.Join(parts, ","))
}

func TestParallelToolCallsInThread(t *testing.T) {
	defer gock.Off()

	httpClient := &http.Client{}
	provider, _ := New(WithBackend(genai.BackendVertexAI), WithLocation("location"), WithProject("project"))
	llm, _ := llmberjack.New(llmberjack.WithDefaultProvider(provider), llmberjack.WithDefaultModel("themodel"), llmberjack.WithHttpClient(httpClient))

	gock.InterceptClient(httpClient)

	const url = "https://location-aiplatform.googleapis.com"
	const path = "/v1beta1/projects/project/locations/location/publishers/google/models/themodel:generateContent"

	var turn2, turn3 []byte

	gock.New(url).Post(path).
		Reply(http.StatusOK).
		SetHeader("content-type", "application/json").
		BodyString(functionCallsResponse([3]string{"call_1", "tool_a", "1"}, [3]string{"call_2", "tool_b", "2"}, [3]string{"call_3", "tool_a", "3"}))

	gock.New(url).Post(path).
		AddMatcher(func(req *http.Request, _ *gock.Request) (bool, error) {
			turn2, _ = io.ReadAll(req.Body)
			return true, nil
		}).
		Reply(http.StatusOK).
		SetHeader("content-type", "application/json").
		BodyString(functionCallsResponse([3]string{"call_4", "tool_b", "4"}, [3]string{"call_5", "tool_a", "5"}))

	gock.New(url).Post(path).
		AddMatcher(func(req *http.Request, _ *gock.Request) (bool, error) {
			turn3, _ = io.ReadAll(req.Body)
			return true, nil
		}).
		Reply(http.StatusOK).
		SetHeader("content-type", "application/json").
		BodyString(aistudioResponse)

	tools := []internal.Tool{echoTool("tool_a"), echoTool("tool_b")}

	resp1, err := llmberjack.NewUntypedRequest().CreateThread().WithTools(tools...).WithText(llmberjack.RoleUser, "start").Do(t.Context(), llm)
	assert.Nil(t, err)

	resp2, err := llmberjack.NewUntypedRequest().FromCandidate(resp1, 0).WithToolExecution(tools...).Do(t.Context(), llm)
	assert.Nil(t, err)

	_, err = llmberjack.NewUntypedRequest().FromCandidate(resp2, 0).WithToolExecution(tools...).Do(t.Context(), llm)
	assert.Nil(t, err)

	assert.False(t, gock.HasUnmatchedRequest())

	// Turn 2: user, model (3 calls), ONE user content with 3 function responses.
	assert.EqualValues(t, 3, gjson.GetBytes(turn2, "contents.#").Int())
	assert.EqualValues(t, 3, gjson.GetBytes(turn2, "contents.1.parts.#(functionCall)#|#").Int())
	assert.Equal(t, "user", gjson.GetBytes(turn2, "contents.2.role").String())
	assert.EqualValues(t, 3, gjson.GetBytes(turn2, "contents.2.parts.#").Int())
	assert.Equal(t, `["call_1","call_2","call_3"]`, gjson.GetBytes(turn2, "contents.2.parts.#.functionResponse.id").Raw)
	assert.Equal(t, `["tool_a","tool_b","tool_a"]`, gjson.GetBytes(turn2, "contents.2.parts.#.functionResponse.name").Raw)
	assert.Equal(t, `["tool_a:1","tool_b:2","tool_a:3"]`, gjson.GetBytes(turn2, "contents.2.parts.#.functionResponse.response.output.text").Raw)

	// Turn 3: the history replays the first batch as a single content, and the
	// second batch, after the intermediate model turn, is a separate content.
	assert.EqualValues(t, 5, gjson.GetBytes(turn3, "contents.#").Int())
	assert.Equal(t, `["call_1","call_2","call_3"]`, gjson.GetBytes(turn3, "contents.2.parts.#.functionResponse.id").Raw)
	assert.Equal(t, "model", gjson.GetBytes(turn3, "contents.3.role").String())
	assert.Equal(t, `["call_4","call_5"]`, gjson.GetBytes(turn3, "contents.3.parts.#.functionCall.id").Raw)
	assert.Equal(t, "user", gjson.GetBytes(turn3, "contents.4.role").String())
	assert.Equal(t, `["call_4","call_5"]`, gjson.GetBytes(turn3, "contents.4.parts.#.functionResponse.id").Raw)
	assert.Equal(t, `["tool_b:4","tool_a:5"]`, gjson.GetBytes(turn3, "contents.4.parts.#.functionResponse.response.output.text").Raw)
}

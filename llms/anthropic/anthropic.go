package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"reflect"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/checkmarble/llmberjack/internal"
	"github.com/cockroachdb/errors"
	"github.com/samber/lo"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
)

type Anthropic struct {
	client  anthropic.Client
	history llmberjack.History[anthropic.MessageParam]

	backend   Backend
	apiKey    string
	baseUrl   string
	project   string
	region    string
	vertexCtx context.Context
	model     *string
}

func (*Anthropic) RequestOptionsType() reflect.Type {
	return reflect.TypeFor[RequestOptions]()
}

func New(opts ...Opt) (*Anthropic, error) {
	llm := Anthropic{
		backend:   BackendAnthropic,
		region:    "global",
		vertexCtx: context.Background(),
	}

	for _, opt := range opts {
		opt(&llm)
	}

	return &llm, nil
}

func (p *Anthropic) Init(adapter internal.Adapter) error {
	switch p.backend {
	case BackendAnthropic:
		return p.initAnthropic(adapter)
	case BackendVertexAI:
		return p.initVertexAI(adapter)
	default:
		return errors.New("unknown backend")
	}
}

func (p *Anthropic) initAnthropic(adapter internal.Adapter) error {
	opts := []option.RequestOption{
		option.WithAPIKey(p.apiKey),
	}

	if adapter.HttpClient() != nil {
		opts = append(opts, option.WithHTTPClient(adapter.HttpClient()))
	}

	if p.baseUrl != "" {
		opts = append(opts, option.WithBaseURL(p.baseUrl))
	}

	client := anthropic.NewClient(opts...)
	p.client = client

	return nil
}

func (p *Anthropic) initVertexAI(adapter internal.Adapter) error {
	// VertexAI requires project and region to be set
	if p.project == "" {
		return errors.New("project is required for VertexAI backend")
	}

	// Create vertex AI client with credentials context
	opts := []option.RequestOption{
		vertex.WithGoogleAuth(p.vertexCtx, p.region, p.project),
	}

	// NB: I'm not passing the adapter's HTTP client to the anthropic client options, because using vertex AI the SDK does automatic
	// application default credentials discovery and generates his own http client. There is scope for a future improvement on this.

	client := anthropic.NewClient(opts...)
	p.client = client

	return nil
}

func (p *Anthropic) ResetThread(threadId *llmberjack.ThreadId) {
	p.history.Clear(threadId)
}

func (p *Anthropic) CopyThread(threadId *llmberjack.ThreadId) *llmberjack.ThreadId {
	return p.history.Copy(threadId)
}

func (p *Anthropic) CloseThread(threadId *llmberjack.ThreadId) {
	p.history.Close(threadId)
}

func (p *Anthropic) ChatCompletion(ctx context.Context, adapter internal.Adapter, requester llmberjack.Requester) (*llmberjack.InnerResponse, error) {
	r := requester.ToRequest()

	model, ok := lo.Coalesce(r.Model, p.model, lo.ToPtr(adapter.DefaultModel()))
	if !ok {
		return nil, errors.New("no model was configured")
	}

	opts := internal.CastProviderOptions[RequestOptions](requester.ProviderRequestOptions(p))

	messages, params, err := p.adaptRequest(adapter, requester, opts)
	if err != nil {
		return nil, errors.Wrap(err, "could not adapt request")
	}

	messageParams := anthropic.MessageNewParams{
		Model:    *model,
		Messages: messages,
	}

	// Add max tokens only if explicitly provided (Vertex AI requires > 0)
	if r.MaxTokens != nil {
		messageParams.MaxTokens = int64(*r.MaxTokens)
	} else if p.backend == BackendVertexAI {
		messageParams.MaxTokens = BackendVertexAiDefaultMaxTokens // Set a default max tokens for Vertex AI if not provided
	}

	// Add system message
	if params.System.Text != "" {
		messageParams.System = []anthropic.TextBlockParam{params.System}
	}

	// Add tools
	if len(params.Tools) > 0 {
		toolUnions := make([]anthropic.ToolUnionParam, len(params.Tools))
		for i, tool := range params.Tools {
			toolUnions[i] = anthropic.ToolUnionParamOfTool(
				tool.InputSchema,
				tool.Name,
			)
			// Description is already set in tool if provided
			toolUnions[i].OfTool.Description = tool.Description
		}
		messageParams.Tools = toolUnions
	}

	// Add tool choice
	if params.ToolChoice != nil {
		messageParams.ToolChoice = *params.ToolChoice
	}

	// Add temperature
	if r.Temperature != nil {
		messageParams.Temperature = anthropic.Float(*r.Temperature)
	}

	// Add top_p
	if r.TopP != nil {
		messageParams.TopP = anthropic.Float(*r.TopP)
	}

	// Add top_k
	if params.TopK != nil {
		messageParams.TopK = anthropic.Int(int64(*params.TopK))
	}

	// Add thinking
	if params.Thinking != nil && *params.Thinking > 0 {
		messageParams.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(*params.Thinking))
	}

	response, err := p.client.Messages.New(ctx, messageParams)
	if err != nil {
		return nil, errors.Wrap(err, "LLM provider failed to generate content")
	}

	responseAdapter, err := p.adaptResponse(adapter, response, requester)
	if err != nil {
		return nil, errors.Wrap(err, "could not adapt response")
	}

	return responseAdapter, nil
}

type requestParams struct {
	System     anthropic.TextBlockParam
	Tools      []anthropic.ToolParam
	ToolChoice *anthropic.ToolChoiceUnionParam
	TopK       *int
	Thinking   *int
}

func (p *Anthropic) adaptRequest(_ internal.Adapter, requester llmberjack.Requester, opts RequestOptions) ([]anthropic.MessageParam, *requestParams, error) {
	r := requester.ToRequest()
	messages := make([]anthropic.MessageParam, 0, len(r.Messages))

	if r.ThreadId != nil {
		messages = append(messages, p.history.Load(r.ThreadId)...)
	}

	params := &requestParams{
		Tools: make([]anthropic.ToolParam, 0, len(r.Tools)),
	}

	// Add tools
	for _, tool := range r.Tools {
		paramsJson, err := json.Marshal(tool.Parameters)
		if err != nil {
			return nil, nil, errors.Wrap(err, "failed to encode tool parameters")
		}

		var toolParams map[string]any
		if err := json.Unmarshal(paramsJson, &toolParams); err != nil {
			return nil, nil, errors.Wrap(err, "failed to decode tool parameters")
		}

		toolParam := anthropic.ToolParam{
			Name: tool.Name,
			InputSchema: anthropic.ToolInputSchemaParam{
				Type:       "object",
				Properties: toolParams,
			},
		}
		if tool.Description != "" {
			toolParam.Description = anthropic.String(tool.Description)
		}
		params.Tools = append(params.Tools, toolParam)
	}

	// System messages are handled as separate messages with RoleSystem

	// Add budget tokens for thinking if specified
	if opts.BudgetTokens != nil && *opts.BudgetTokens > 0 {
		params.Thinking = opts.BudgetTokens
	}

	// Process messages
	for _, msg := range r.Messages {
		parts := make([]anthropic.ContentBlockParamUnion, 0, len(msg.Parts))

		for _, part := range msg.Parts {
			if seeker, ok := part.(io.ReadSeeker); ok {
				if _, err := seeker.Seek(0, io.SeekStart); err != nil {
					return nil, nil, err
				}
			}

			buf, err := io.ReadAll(part)
			if err != nil {
				return nil, nil, errors.Wrap(err, "could not read content part")
			}

			switch msg.Type {
			case llmberjack.TypeText:
				parts = append(parts, anthropic.NewTextBlock(string(buf)))
			}
		}

		var msgParam anthropic.MessageParam

		switch msg.Role {
		case llmberjack.RoleAi:
			msgParam = anthropic.MessageParam{
				Role:    anthropic.MessageParamRoleAssistant,
				Content: parts,
			}

		case llmberjack.RoleUser:
			msgParam = anthropic.MessageParam{
				Role:    anthropic.MessageParamRoleUser,
				Content: parts,
			}

		case llmberjack.RoleSystem:
			// System messages are handled separately in Anthropic SDK
			continue

		case llmberjack.RoleTool:
			if msg.Tool == nil {
				return nil, nil, errors.New("sent a tool response when no tool was invoked")
			}

			var toolContent string
			for _, part := range msg.Parts {
				if seeker, ok := part.(io.ReadSeeker); ok {
					_, err := seeker.Seek(0, io.SeekStart)
					if err != nil {
						return nil, nil, errors.Wrap(err, "could not seek content part")
					}
				}
				buf, err := io.ReadAll(part)
				if err != nil {
					return nil, nil, errors.Wrap(err, "could not read content part")
				}
				toolContent = string(buf)
			}

			msgParam = anthropic.MessageParam{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					anthropic.NewToolResultBlock(msg.Tool.Id, toolContent, false),
				},
			}
		}

		if r.ThreadId != nil && !r.SkipSaveInput {
			p.history.Save(r.ThreadId, msgParam)
		}

		messages = append(messages, msgParam)
	}

	return messages, params, nil
}

func (p *Anthropic) adaptResponse(_ internal.Adapter, response *anthropic.Message, requester llmberjack.Requester) (*llmberjack.InnerResponse, error) {
	resp := llmberjack.InnerResponse{
		Id:         response.ID,
		Model:      response.Model,
		Candidates: make([]llmberjack.ResponseCandidate, 1),
	}

	var finishReason llmberjack.FinishReason
	switch response.StopReason {
	case "end_turn":
		finishReason = llmberjack.FinishReasonStop
	case "max_tokens":
		finishReason = llmberjack.FinishReasonMaxTokens
	default:
		finishReason = llmberjack.FinishReason(response.StopReason)
	}

	text := ""
	toolCalls := make([]llmberjack.ResponseToolCall, 0)
	thoughts := ""

	for _, content := range response.Content {
		switch content.Type {
		case "text":
			textBlock := content.AsText()
			text += textBlock.Text
		case "tool_use":
			toolBlock := content.AsToolUse()
			toolCalls = append(toolCalls, llmberjack.ResponseToolCall{
				Id:         toolBlock.ID,
				Name:       toolBlock.Name,
				Parameters: toolBlock.Input,
			})
		case "thinking":
			thinkingBlock := content.AsThinking()
			thoughts = thinkingBlock.Thinking
		}
	}

	resp.Candidates[0] = llmberjack.ResponseCandidate{
		Text:         text,
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		Thoughts:     thoughts,
		SelectCandidate: func() {
			req := requester.ToRequest()

			if req.ThreadId != nil && !req.SkipSaveOutput {
				msg := anthropic.MessageParam{
					Role:    anthropic.MessageParamRoleAssistant,
					Content: []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock(text)},
				}

				p.history.Save(req.ThreadId, msg)
			}
		},
	}

	return &resp, nil
}

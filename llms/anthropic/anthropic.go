package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"reflect"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/checkmarble/llmberjack/internal"
	"github.com/cockroachdb/errors"
	"github.com/invopop/jsonschema"
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

	params, err := p.adaptRequest(adapter, requester, *model, opts)
	if err != nil {
		return nil, errors.Wrap(err, "could not adapt request")
	}

	response, err := p.client.Messages.New(ctx, *params)
	if err != nil {
		return nil, errors.Wrap(err, "LLM provider failed to generate content")
	}

	responseAdapter, err := p.adaptResponse(adapter, response, requester)
	if err != nil {
		return nil, errors.Wrap(err, "could not adapt response")
	}

	return responseAdapter, nil
}

func (p *Anthropic) adaptRequest(_ internal.Adapter, requester llmberjack.Requester, model string, opts RequestOptions) (*anthropic.MessageNewParams, error) {
	r := requester.ToRequest()
	messages := make([]anthropic.MessageParam, 0, len(r.Messages))

	if r.ThreadId != nil {
		messages = append(messages, p.history.Load(r.ThreadId)...)
	}

	// Build tools
	tools := make([]anthropic.ToolParam, 0, len(r.Tools))
	for _, tool := range r.Tools {
		inputSchema, err := adaptToolInputSchema(tool.Parameters)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to adapt parameters of tool '%s'", tool.Name)
		}

		toolParam := anthropic.ToolParam{
			Name:        tool.Name,
			InputSchema: inputSchema,
		}
		if tool.Description != "" {
			toolParam.Description = anthropic.String(tool.Description)
		}
		tools = append(tools, toolParam)
	}

	// Handle system messages
	var systemText string
	for _, msg := range r.Messages {
		if msg.Role == llmberjack.RoleSystem {
			for _, part := range msg.Parts {
				if seeker, ok := part.(io.ReadSeeker); ok {
					_, err := seeker.Seek(0, io.SeekStart)
					if err != nil {
						return nil, errors.Wrap(err, "could not seek content part")
					}
				}
				buf, _ := io.ReadAll(part)
				if systemText != "" {
					systemText += "\n"
				}
				systemText += string(buf)
			}
		}
	}

	// Anthropic expects all the tool_result blocks answering an assistant turn
	// (which may contain several parallel tool_use blocks) to be in the single
	// user message following it. Consecutive tool response messages are
	// therefore accumulated here, and only emitted (and saved to history) once
	// another message is encountered, or at the end of the request messages.
	var pendingToolResults []anthropic.ContentBlockParamUnion

	flushToolResults := func() {
		if len(pendingToolResults) == 0 {
			return
		}

		msgParam := anthropic.MessageParam{
			Role:    anthropic.MessageParamRoleUser,
			Content: pendingToolResults,
		}

		if r.ThreadId != nil && !r.SkipSaveInput {
			p.history.Save(r.ThreadId, msgParam)
		}

		messages = append(messages, msgParam)
		pendingToolResults = nil
	}

	// Process messages
	for _, msg := range r.Messages {
		if msg.Role == llmberjack.RoleSystem {
			continue // Already handled above
		}

		if msg.Role != llmberjack.RoleTool {
			flushToolResults()
		}

		parts := make([]anthropic.ContentBlockParamUnion, 0, len(msg.Parts))

		for _, part := range msg.Parts {
			if seeker, ok := part.(io.ReadSeeker); ok {
				if _, err := seeker.Seek(0, io.SeekStart); err != nil {
					return nil, err
				}
			}

			buf, err := io.ReadAll(part)
			if err != nil {
				return nil, errors.Wrap(err, "could not read content part")
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

		case llmberjack.RoleTool:
			if msg.Tool == nil {
				return nil, errors.New("sent a tool response when no tool was invoked")
			}

			var toolContent string
			for _, part := range msg.Parts {
				if seeker, ok := part.(io.ReadSeeker); ok {
					_, err := seeker.Seek(0, io.SeekStart)
					if err != nil {
						return nil, errors.Wrap(err, "could not seek content part")
					}
				}
				buf, err := io.ReadAll(part)
				if err != nil {
					return nil, errors.Wrap(err, "could not read content part")
				}
				toolContent = string(buf)
			}

			pendingToolResults = append(pendingToolResults, anthropic.NewToolResultBlock(msg.Tool.Id, toolContent, false))

			continue
		}

		if r.ThreadId != nil && !r.SkipSaveInput {
			p.history.Save(r.ThreadId, msgParam)
		}

		messages = append(messages, msgParam)
	}

	flushToolResults()

	// Build the complete MessageNewParams
	params := &anthropic.MessageNewParams{
		Model:    model,
		Messages: messages,
	}
	if r.PromptCaching != nil && *r.PromptCaching {
		params.CacheControl = anthropic.NewCacheControlEphemeralParam()
	}

	// Add system message if present
	if systemText != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: systemText},
		}
	}

	// Add tools if present
	if len(tools) > 0 {
		toolUnions := make([]anthropic.ToolUnionParam, len(tools))
		for i, tool := range tools {
			toolUnions[i] = anthropic.ToolUnionParamOfTool(
				tool.InputSchema,
				tool.Name,
			)
			toolUnions[i].OfTool.Description = tool.Description
		}
		params.Tools = toolUnions
	}

	// max_tokens is required by the API on every backend, so fall back to the default
	// when the request does not set one.
	if r.MaxTokens != nil {
		params.MaxTokens = int64(*r.MaxTokens)
	} else {
		params.MaxTokens = DefaultMaxTokens
	}

	// Add temperature
	if r.Temperature != nil {
		params.Temperature = anthropic.Float(*r.Temperature)
	}

	// Add top_p
	if r.TopP != nil {
		params.TopP = anthropic.Float(*r.TopP)
	}

	// Resolve thinking configuration
	if r.Thinking != nil && !*r.Thinking {
		// Explicitly disabled, do not set thinking
		disabled := anthropic.NewThinkingConfigDisabledParam()
		params.Thinking = anthropic.ThinkingConfigParamUnion{OfDisabled: &disabled}
	} else if r.ThinkingLevel != nil {
		adaptive := anthropic.ThinkingConfigAdaptiveParam{Display: "summarized", Type: "adaptive"}
		params.Thinking = anthropic.ThinkingConfigParamUnion{OfAdaptive: &adaptive}
		params.OutputConfig.Effort = adaptThinkingLevel(*r.ThinkingLevel)
	} else if opts.BudgetTokens != nil && *opts.BudgetTokens > 0 {
		params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(*opts.BudgetTokens))
	} else if r.Thinking != nil && *r.Thinking {
		adaptive := anthropic.ThinkingConfigAdaptiveParam{Display: "summarized", Type: "adaptive"}
		params.Thinking = anthropic.ThinkingConfigParamUnion{OfAdaptive: &adaptive}
	}

	return params, nil
}

// adaptToolInputSchema converts the JSON schema generated for a tool's
// arguments into Anthropic's tool input schema.
//
// The generated schema is a full object schema ({"$schema", "type",
// "properties", "required", ...}), so its "properties" and "required" are
// mapped to the matching fields of the input schema. Other keywords (such as
// "additionalProperties" or "$defs" referenced by nested types) are kept at the
// root of the input schema, and "$schema" is dropped since Anthropic rejects
// it.
func adaptToolInputSchema(schema jsonschema.Schema) (anthropic.ToolInputSchemaParam, error) {
	schemaJson, err := json.Marshal(schema)
	if err != nil {
		return anthropic.ToolInputSchemaParam{}, errors.Wrap(err, "failed to encode tool parameters")
	}

	var fields map[string]any
	if err := json.Unmarshal(schemaJson, &fields); err != nil {
		return anthropic.ToolInputSchemaParam{}, errors.Wrap(err, "failed to decode tool parameters")
	}

	inputSchema := anthropic.ToolInputSchemaParam{
		Type:       "object",
		Properties: fields["properties"],
	}

	if required, ok := fields["required"].([]any); ok {
		inputSchema.Required = make([]string, 0, len(required))

		for _, field := range required {
			if name, ok := field.(string); ok {
				inputSchema.Required = append(inputSchema.Required, name)
			}
		}
	}

	delete(fields, "$schema")
	delete(fields, "type")
	delete(fields, "properties")
	delete(fields, "required")

	if len(fields) > 0 {
		inputSchema.ExtraFields = fields
	}

	return inputSchema, nil
}

func adaptThinkingLevel(level llmberjack.ThinkingLevel) anthropic.OutputConfigEffort {
	switch level {
	case llmberjack.ThinkingLevelLow:
		return anthropic.OutputConfigEffortLow
	case llmberjack.ThinkingLevelMedium:
		return anthropic.OutputConfigEffortMedium
	case llmberjack.ThinkingLevelHigh:
		return anthropic.OutputConfigEffortHigh
	default:
		return ""
	}
}

func (p *Anthropic) adaptResponse(_ internal.Adapter, response *anthropic.Message, requester llmberjack.Requester) (*llmberjack.InnerResponse, error) {
	resp := llmberjack.InnerResponse{
		Id:         response.ID,
		Model:      response.Model,
		Candidates: make([]llmberjack.ResponseCandidate, 1),
		Usage: llmberjack.ResponseUsage{
			InputTokens: response.Usage.InputTokens +
				response.Usage.CacheCreationInputTokens +
				response.Usage.CacheReadInputTokens,
			OutputTokens:     response.Usage.OutputTokens,
			CacheWriteTokens: response.Usage.CacheCreationInputTokens,
			CacheReadTokens:  response.Usage.CacheReadInputTokens,
		},
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
				// Save the whole assistant message, not only its text: the
				// tool_use blocks must be in the history for the following
				// tool_result blocks to be accepted, and thinking blocks (with
				// their signatures) must be sent back when using tools.
				p.history.Save(req.ThreadId, response.ToParam())
			}
		},
	}

	return &resp, nil
}

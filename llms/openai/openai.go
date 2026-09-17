package openai

import (
	"context"
	"encoding/json"
	"io"
	"reflect"
	"time"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/checkmarble/llmberjack/internal"
	"github.com/cockroachdb/errors"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/samber/lo"
)

type OpenAi struct {
	client  openai.Client
	history llmberjack.History[openai.ChatCompletionMessageParamUnion]

	RequestHookFunc  func(llmberjack.Requester, *openai.ChatCompletionNewParams) error
	ResponseHookFunc func(*openai.ChatCompletion, *llmberjack.InnerResponse) error

	baseUrl string
	apiKey  string
	model   *string
}

func (*OpenAi) RequestOptionsType() reflect.Type {
	return nil
}

func New(opts ...Opt) (*OpenAi, error) {
	llm := OpenAi{}

	for _, opt := range opts {
		opt(&llm)
	}

	return &llm, nil
}

func (p *OpenAi) Init(llm internal.Adapter) error {
	opts := []option.RequestOption{
		option.WithAPIKey(p.apiKey),
	}

	if llm.HttpClient() != nil {
		opts = append(opts, option.WithHTTPClient(llm.HttpClient()))
	}
	if p.baseUrl != "" {
		opts = append(opts, option.WithBaseURL(p.baseUrl))
	}

	p.client = openai.NewClient(opts...)

	return nil
}

func (p *OpenAi) ResetThread(threadId *llmberjack.ThreadId) {
	p.history.Clear(threadId)
}

func (p *OpenAi) CopyThread(threadId *llmberjack.ThreadId) *llmberjack.ThreadId {
	return p.history.Copy(threadId)
}

func (p *OpenAi) CloseThread(threadId *llmberjack.ThreadId) {
	p.history.Close(threadId)
}

func (p *OpenAi) ChatCompletion(ctx context.Context, llm internal.Adapter, requester llmberjack.Requester) (*llmberjack.InnerResponse, error) {
	cfg, err := p.adaptRequest(llm, requester)
	if err != nil {
		return nil, errors.Wrap(err, "could not adapt request")
	}

	if p.RequestHookFunc != nil {
		if err := p.RequestHookFunc(requester, cfg); err != nil {
			return nil, err
		}
	}

	response, err := p.client.Chat.Completions.New(ctx, *cfg)
	if err != nil {
		return nil, errors.Wrap(err, "LLM provider failed to generate content")
	}

	responseAdapter, err := p.adaptResponse(llm, response, requester)
	if err != nil {
		return nil, errors.Wrap(err, "could not adapt response")
	}

	if p.ResponseHookFunc != nil {
		if err := p.ResponseHookFunc(response, responseAdapter); err != nil {
			return nil, err
		}
	}

	return responseAdapter, nil
}

func (p *OpenAi) adaptRequest(llm internal.Adapter, requester llmberjack.Requester) (*openai.ChatCompletionNewParams, error) {
	r := requester.ToRequest()
	contents := make([]openai.ChatCompletionMessageParamUnion, 0, len(r.Messages))

	if r.ThreadId != nil {
		contents = append(contents, p.history.Load(r.ThreadId)...)
	}

	model, ok := lo.Coalesce(r.Model, p.model, lo.ToPtr(llm.DefaultModel()))
	if !ok {
		return nil, errors.New("no model was configured")
	}

	cfg := openai.ChatCompletionNewParams{
		Model:    *model,
		Messages: contents,
	}

	if r.MaxCandidates != nil {
		cfg.N = openai.Int(int64(*r.MaxCandidates))
	}
	if r.MaxTokens != nil {
		cfg.MaxTokens = openai.Int(int64(*r.MaxTokens))
	}
	if r.Temperature != nil {
		cfg.Temperature = openai.Float(*r.Temperature)
	}
	if r.TopP != nil {
		cfg.TopP = openai.Float(*r.TopP)
	}
	if r.ThinkingLevel != nil && (r.Thinking == nil || *r.Thinking) {
		cfg.ReasoningEffort = adaptThinkingLevel(*r.ThinkingLevel)
	}

	if r.ResponseSchema != nil {
		cfg.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
				JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:        r.SchemaName,
					Description: openai.String(r.SchemaDescription),
					Schema:      lo.CoalesceOrEmpty(r.SchemaOverride, r.ResponseSchema),
					Strict:      openai.Bool(true),
				},
			},
		}
	}

	for _, tool := range r.Tools {
		paramsJson, err := json.Marshal(tool.Parameters)
		if err != nil {
			return nil, errors.Wrap(err, "failed to encode tool parameters")
		}

		var params map[string]any

		if err := json.Unmarshal(paramsJson, &params); err != nil {
			return nil, errors.Wrap(err, "failed to encode tool parameters")
		}

		cfg.Tools = append(cfg.Tools, openai.ChatCompletionToolParam{
			Type: "function",
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: openai.String(tool.Description),
				Parameters:  openai.FunctionParameters(params),
			},
		})
	}

	for _, msg := range r.Messages {
		parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(msg.Parts))

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
				parts = append(parts, openai.TextContentPart(string(buf)))
			}
		}

		content := openai.ChatCompletionMessageParamUnion{}

		switch msg.Role {
		case llmberjack.RoleAi:
			content.OfAssistant = &openai.ChatCompletionAssistantMessageParam{
				Content: openai.ChatCompletionAssistantMessageParamContentUnion{
					OfArrayOfContentParts: lo.Map(parts, func(p openai.ChatCompletionContentPartUnionParam, _ int) openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion {
						return openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
							OfText: &openai.ChatCompletionContentPartTextParam{
								Text: *p.GetText(),
							},
						}
					}),
				},
			}

		case llmberjack.RoleUser:
			content.OfUser = &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfArrayOfContentParts: lo.Map(parts, func(p openai.ChatCompletionContentPartUnionParam, _ int) openai.ChatCompletionContentPartUnionParam {
						return openai.ChatCompletionContentPartUnionParam{
							OfText: &openai.ChatCompletionContentPartTextParam{
								Text: *p.GetText(),
							},
						}
					}),
				},
			}

		case llmberjack.RoleSystem:
			content.OfSystem = &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfArrayOfContentParts: lo.Map(parts, func(p openai.ChatCompletionContentPartUnionParam, _ int) openai.ChatCompletionContentPartTextParam {
						return openai.ChatCompletionContentPartTextParam{
							Text: *p.GetText(),
						}
					}),
				},
			}

		case llmberjack.RoleTool:
			content.OfTool = &openai.ChatCompletionToolMessageParam{
				ToolCallID: msg.Tool.Id,
				Content: openai.ChatCompletionToolMessageParamContentUnion{
					OfArrayOfContentParts: lo.Map(parts, func(p openai.ChatCompletionContentPartUnionParam, _ int) openai.ChatCompletionContentPartTextParam {
						return openai.ChatCompletionContentPartTextParam{
							Text: *p.GetText(),
						}
					}),
				},
			}
		}

		if r.ThreadId != nil && !r.SkipSaveInput {
			p.history.Save(r.ThreadId, content)
		}

		cfg.Messages = append(cfg.Messages, content)
	}

	return &cfg, nil
}

func adaptThinkingLevel(level llmberjack.ThinkingLevel) openai.ReasoningEffort {
	switch level {
	case llmberjack.ThinkingLevelLow:
		return openai.ReasoningEffortLow
	case llmberjack.ThinkingLevelMedium:
		return openai.ReasoningEffortMedium
	case llmberjack.ThinkingLevelHigh:
		return openai.ReasoningEffortHigh
	default:
		return ""
	}
}

func (p *OpenAi) adaptResponse(_ internal.Adapter, response *openai.ChatCompletion, requester llmberjack.Requester) (*llmberjack.InnerResponse, error) {
	resp := llmberjack.InnerResponse{
		Id:         response.ID,
		Model:      response.Model,
		Candidates: make([]llmberjack.ResponseCandidate, len(response.Choices)),
		Created:    time.Unix(response.Created, 0),
		Usage: llmberjack.ResponseUsage{
			InputTokens:     response.Usage.PromptTokens,
			OutputTokens:    response.Usage.CompletionTokens,
			CacheReadTokens: response.Usage.PromptTokensDetails.CachedTokens,
		},
	}

	for idx, candidate := range response.Choices {
		var finishReason llmberjack.FinishReason

		switch candidate.FinishReason {
		case "stop":
			finishReason = llmberjack.FinishReasonStop
		case "length":
			finishReason = llmberjack.FinishReasonMaxTokens
		case "content_filter":
			finishReason = llmberjack.FinishReasonContentFilter
		default:
			finishReason = llmberjack.FinishReason(candidate.FinishReason)
		}

		toolCalls := make([]llmberjack.ResponseToolCall, len(candidate.Message.ToolCalls))

		for idx, toolCall := range candidate.Message.ToolCalls {
			toolCalls[idx] = llmberjack.ResponseToolCall{
				Id:         toolCall.ID,
				Name:       toolCall.Function.Name,
				Parameters: []byte(toolCall.Function.Arguments),
			}
		}

		resp.Candidates[idx] = llmberjack.ResponseCandidate{
			Text:         candidate.Message.Content,
			ToolCalls:    toolCalls,
			FinishReason: finishReason,
			SelectCandidate: func() {
				req := requester.ToRequest()

				if req.ThreadId != nil && !req.SkipSaveOutput {
					msg := openai.ChatCompletionMessageParamUnion{
						OfAssistant: &openai.ChatCompletionAssistantMessageParam{
							ToolCalls: candidate.Message.ToParam().GetToolCalls(),
							Content: openai.ChatCompletionAssistantMessageParamContentUnion{
								OfString: openai.String(candidate.Message.Content),
							},
						},
					}

					p.history.Save(req.ThreadId, msg)
				}
			},
		}
	}

	return &resp, nil
}

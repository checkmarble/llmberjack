package aistudio

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"reflect"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/checkmarble/llmberjack/internal"
	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"google.golang.org/genai"
)

type AiStudio struct {
	client  *genai.Client
	history llmberjack.History[*genai.Content]

	backend  genai.Backend
	apiKey   string
	project  string
	location string
	model    *string
}

func (*AiStudio) RequestOptionsType() reflect.Type {
	return reflect.TypeFor[RequestOptions]()
}

func New(opts ...Opt) (*AiStudio, error) {
	llm := AiStudio{
		backend: genai.BackendGeminiAPI,
	}

	for _, opt := range opts {
		opt(&llm)
	}

	return &llm, nil
}

func (p *AiStudio) Init(adapter internal.Adapter) error {
	cfg := genai.ClientConfig{
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: adapter.HttpClient(),
	}

	if p.backend != genai.BackendUnspecified {
		cfg.Backend = p.backend
	}
	switch cfg.Backend {
	case genai.BackendGeminiAPI:
		cfg.APIKey = p.apiKey
	case genai.BackendVertexAI:
		cfg.Project = p.project

		cfg.Location = p.location

		switch p.location {
		case "eu", "us":
			cfg.HTTPOptions.BaseURL = fmt.Sprintf("https://aiplatform.%s.rep.googleapis.com", p.location)
		}
	}

	client, err := genai.NewClient(context.Background(), &cfg)
	if err != nil {
		return err
	}

	p.client = client

	return nil
}

func (p *AiStudio) ResetThread(threadId *llmberjack.ThreadId) {
	p.history.Clear(threadId)
}

func (p *AiStudio) CopyThread(threadId *llmberjack.ThreadId) *llmberjack.ThreadId {
	return p.history.Copy(threadId)
}

func (p *AiStudio) CloseThread(threadId *llmberjack.ThreadId) {
	p.history.Close(threadId)
}

func (p *AiStudio) ChatCompletion(ctx context.Context, llm internal.Adapter, requester llmberjack.Requester) (*llmberjack.InnerResponse, error) {
	model, ok := lo.Coalesce(requester.ToRequest().Model, p.model, lo.ToPtr(llm.DefaultModel()))
	if !ok {
		return nil, errors.New("no model was configured")
	}

	opts := internal.CastProviderOptions[RequestOptions](requester.ProviderRequestOptions(p))

	contents, cfg, err := p.adaptRequest(llm, requester, opts)
	if err != nil {
		return nil, errors.Wrap(err, "could not adapt request")
	}
	response, err := p.client.Models.GenerateContent(ctx, *model, contents, cfg)
	if err != nil {
		return nil, errors.Wrap(err, "LLM provider failed to generate content")
	}

	return p.adaptResponse(llm, response, requester)
}

func (p *AiStudio) adaptRequest(_ internal.Adapter, requester llmberjack.Requester, opts RequestOptions) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	r := requester.ToRequest()
	contents := make([]*genai.Content, 0, len(r.Messages))

	if r.ThreadId != nil {
		contents = append(contents, p.history.Load(r.ThreadId)...)
	}

	cfg := genai.GenerateContentConfig{
		CandidateCount:  int32(lo.FromPtr(r.MaxCandidates)),
		MaxOutputTokens: int32(lo.FromPtr(r.MaxTokens)),
		Temperature:     internal.MaybeF64ToF32(r.Temperature),
		TopP:            internal.MaybeF64ToF32(r.TopP),
		TopK:            internal.MaybeF64ToF32(opts.TopK),
	}

	if lo.FromPtr(opts.GoogleSearch) {
		cfg.Tools = []*genai.Tool{{
			GoogleSearch: &genai.GoogleSearch{},
		}}
	}

	if r.Thinking != nil && !lo.FromPtr(r.Thinking) {
		cfg.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingLevel: genai.ThinkingLevelLow,
		}
	} else if opts.Thinking != nil || r.ThinkingLevel != nil {
		thinkingConfig := genai.ThinkingConfig{}
		if opts.Thinking != nil {
			thinkingConfig.IncludeThoughts = opts.Thinking.IncludeThoughts
			thinkingConfig.ThinkingLevel = genai.ThinkingLevel(opts.Thinking.ThinkingLevel)
		}
		if r.ThinkingLevel != nil {
			thinkingConfig.ThinkingLevel = adaptThinkingLevel(*r.ThinkingLevel)
		}
		cfg.ThinkingConfig = &thinkingConfig
	}

	if r.ResponseSchema != nil {
		r.ResponseSchema.Description = r.SchemaDescription

		if r.SchemaOverride != nil {
			r.SchemaOverride.Description = r.SchemaDescription
		}

		cfg.ResponseMIMEType = "application/json"
		cfg.ResponseJsonSchema = lo.CoalesceOrEmpty(r.SchemaOverride, r.ResponseSchema)
	}

	cfg.Tools = append(cfg.Tools, lo.MapToSlice(r.Tools, func(_ string, t internal.Tool) *genai.Tool {
		return &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:                 t.Name,
					Description:          t.Description,
					ParametersJsonSchema: t.Parameters,
				},
			},
		}
	})...)

Messages:
	for _, msg := range r.Messages {
		parts := make([]*genai.Part, 0, len(msg.Parts))

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
				parts = append(parts, genai.NewPartFromText(string(buf)))
			}
		}

		role := genai.RoleUser

		switch msg.Role {
		case llmberjack.RoleAi:
			role = genai.RoleModel
		case llmberjack.RoleUser:
			role = genai.RoleUser
		case llmberjack.RoleTool:
			if msg.Tool == nil {
				return nil, nil, errors.New("sent a tool response when no tool was invoked")
			}

			msg := &genai.Content{
				Role: role,
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       msg.Tool.Id,
							Name:     msg.Tool.Name,
							Response: map[string]any{"output": parts[0]},
						},
					},
				},
			}

			contents = append(contents, msg)

			if r.ThreadId != nil && !r.SkipSaveInput {
				p.history.Save(r.ThreadId, msg)
			}

			continue Messages
		case llmberjack.RoleSystem:
			if cfg.SystemInstruction == nil {
				cfg.SystemInstruction = &genai.Content{
					Parts: make([]*genai.Part, 0),
				}
			}

			cfg.SystemInstruction.Parts = append(cfg.SystemInstruction.Parts, parts...)

			if r.ThreadId != nil && !r.SkipSaveInput {
				p.history.Save(r.ThreadId, cfg.SystemInstruction)
			}

			continue Messages
		}

		content := &genai.Content{
			Role:  role,
			Parts: parts,
		}

		if r.ThreadId != nil && !r.SkipSaveInput {
			p.history.Save(r.ThreadId, content)
		}

		contents = append(contents, content)
	}

	return contents, &cfg, nil
}

func adaptThinkingLevel(level llmberjack.ThinkingLevel) genai.ThinkingLevel {
	switch level {
	case llmberjack.ThinkingLevelLow:
		return genai.ThinkingLevelLow
	case llmberjack.ThinkingLevelMedium:
		return genai.ThinkingLevelMedium
	case llmberjack.ThinkingLevelHigh:
		return genai.ThinkingLevelHigh
	default:
		return genai.ThinkingLevelUnspecified
	}
}

func (p *AiStudio) adaptResponse(_ internal.Adapter, response *genai.GenerateContentResponse, requester llmberjack.Requester) (*llmberjack.InnerResponse, error) {
	resp := llmberjack.InnerResponse{
		Id:         response.ResponseID,
		Model:      response.ModelVersion,
		Candidates: make([]llmberjack.ResponseCandidate, len(response.Candidates)),
		Created:    response.CreateTime,
	}
	if response.UsageMetadata != nil {
		resp.Usage = llmberjack.ResponseUsage{
			InputTokens: int64(response.UsageMetadata.PromptTokenCount +
				response.UsageMetadata.ToolUsePromptTokenCount),
			OutputTokens: int64(response.UsageMetadata.CandidatesTokenCount +
				response.UsageMetadata.ThoughtsTokenCount),
			CacheReadTokens: int64(response.UsageMetadata.CachedContentTokenCount),
		}
	}

	for idx, candidate := range response.Candidates {
		if len(candidate.Content.Parts) == 0 {
			return nil, errors.New("LLM provider generated no content")
		}

		var finishReason llmberjack.FinishReason

		switch candidate.FinishReason {
		case genai.FinishReasonStop:
			finishReason = llmberjack.FinishReasonStop
		case genai.FinishReasonMaxTokens:
			finishReason = llmberjack.FinishReasonMaxTokens
		case genai.FinishReasonProhibitedContent:
			finishReason = llmberjack.FinishReasonContentFilter
		default:
			finishReason = llmberjack.FinishReason(candidate.FinishReason)
		}

		var grounding *llmberjack.ResponseGrounding

		if candidate.GroundingMetadata != nil {
			grounding = &llmberjack.ResponseGrounding{
				Searches: candidate.GroundingMetadata.WebSearchQueries,
				Sources: lo.Map(candidate.GroundingMetadata.GroundingChunks, func(c *genai.GroundingChunk, _ int) llmberjack.ResponseGroundingSource {
					return llmberjack.ResponseGroundingSource{
						Domain: lo.CoalesceOrEmpty(c.Web.Domain, c.Web.Title),
						Url:    c.Web.URI,
					}
				}),
				Snippets: lo.Map(candidate.GroundingMetadata.GroundingSupports, func(s *genai.GroundingSupport, _ int) string {
					return s.Segment.Text
				}),
			}
		}

		toolCalls := make([]llmberjack.ResponseToolCall, len(response.FunctionCalls()))

		for idx, toolCall := range response.FunctionCalls() {
			params, err := json.Marshal(toolCall.Args)
			if err != nil {
				return nil, errors.Wrap(err, "failed to parse tool call parameters")
			}

			toolCalls[idx] = llmberjack.ResponseToolCall{
				Id:         toolCall.ID,
				Name:       toolCall.Name,
				Parameters: params,
			}
		}

		resp.Candidates[idx] = llmberjack.ResponseCandidate{
			Text:         candidate.Content.Parts[0].Text,
			ToolCalls:    toolCalls,
			FinishReason: finishReason,
			Grounding:    grounding,
			SelectCandidate: func() {
				req := requester.ToRequest()

				if req.ThreadId != nil && !req.SkipSaveOutput {
					p.history.Save(req.ThreadId, candidate.Content)
				}
			},
		}
	}

	return &resp, nil
}

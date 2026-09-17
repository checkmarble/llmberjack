package llmberjack

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"reflect"
	"strings"

	"github.com/checkmarble/llmberjack/internal"
	"github.com/cockroachdb/errors"
	"github.com/invopop/jsonschema"
	"github.com/samber/lo"
)

type (
	MessageRole int
	MessageType int
)

const (
	RoleSystem MessageRole = iota
	RoleUser
	RoleAi
	RoleTool
)

const (
	TypeText MessageType = iota
)

// Requester represents something that can be turned into a request.
//
// Used internally to abstract over request types across packages.
type Requester interface {
	// ToRequest unwraps the actual request.
	ToRequest() innerRequest
	// ProviderRequestOptions extracts the provider-specific configuration
	// options for a given provider. This is called from each provider to
	// retrieve its specific configuration in a type-safe manner.
	ProviderRequestOptions(provider Llm) internal.ProviderRequestOptions
}

// Message is an abstraction over a "prompt".
type Message struct {
	// Type is the binary representation of the message
	Type MessageType
	// Role represent "who" (or "what") composed a message. Note that all
	// provider will not support all of the roles, but must still account for
	// them.
	Role MessageRole
	// Parts are subdivision of a specific message.
	Parts []io.Reader

	// Tool is an instruction from a tool function to be called. This only makes
	// sense in response messages.
	Tool *ResponseToolCall
}

// innerRequest represents the actual request to be sent to the provider, before
// being adapted for it.
type innerRequest struct {
	ThreadId       *ThreadId
	SkipSaveInput  bool
	SkipSaveOutput bool

	Model          *string
	ModelFunc      func(llm Llm, providerName *string) string
	Messages       []Message
	ResponseSchema *jsonschema.Schema
	Tools          map[string]internal.Tool

	SchemaName        string
	SchemaDescription string
	SchemaOverride    *jsonschema.Schema

	MaxTokens     *int
	MaxCandidates *int
	Temperature   *float64
	TopP          *float64

	// Thinking is a flag to enable/disable thinking. If not provided, the provider will use its default behavior.
	Thinking *bool
	// ThinkingLevel controls the amount of reasoning used by providers that support it.
	ThinkingLevel *ThinkingLevel
	// PromptCaching enables provider-managed prompt caching when explicit opt-in is required.
	// Providers with implicit caching may ignore this setting.
	PromptCaching *bool

	ProviderOptions map[reflect.Type]internal.ProviderRequestOptions
}

// ThinkingLevel is a portable reasoning effort level.
type ThinkingLevel string

const (
	ThinkingLevelLow    ThinkingLevel = "low"
	ThinkingLevelMedium ThinkingLevel = "medium"
	ThinkingLevelHigh   ThinkingLevel = "high"
)

// Request represent a request to be sent the a provider, in the context of the
// current conversation.
//
// It contains an `innerRequest` built by the caller, but also optionally tracks
// which candidate it responds to, in order to link tool responses to their
// corresponding tool calls.
//
// It is generic in T which it will use to unmarshal the response into a typed
// struct.
type Request[T any] struct {
	innerRequest

	provider        *string
	createNewThread bool
	respondsTo      *ResponseCandidate
	err             error
}

// NewUntypedRequest is a helper method to create a `Request` which will be a
// raw string, without unmarshalling the response into a struct.
func NewUntypedRequest() Request[string] {
	return Request[string]{
		innerRequest: innerRequest{
			Tools:           make(map[string]internal.Tool),
			ProviderOptions: make(map[reflect.Type]internal.ProviderRequestOptions),
		},
	}
}

// NewRequest creates a builder to craft a request to sent to an LLM provider.
//
// It provides a series of methods to chain-call in order to add context,
// prompts and configuration.
//
// It is generic in T, which will be used to generate a JSONSchema to be used as
// a response schema in the request. See [this](https://github.com/invopop/jsonschema)
// for more information about how to write the structs.
//
// Example usage:
//
//	resp, err := llmberjack.NewRequest[Output]().
//		WithText(llmberjack.RoleUser, "How are you today?").
//		Do(ctx, llm)
func NewRequest[T any]() Request[T] {
	r := innerRequest{
		Tools:           make(map[string]internal.Tool),
		ProviderOptions: make(map[reflect.Type]internal.ProviderRequestOptions),
	}

	switch any(*new(T)).(type) {
	case string:
	default:
		r.ResponseSchema = lo.ToPtr(internal.GenerateSchema[T]())
	}

	return Request[T]{
		innerRequest: r,
	}
}

// Do executes a built request on the configured provider.
//
// It will return a response generic over the configured typed on the Request,
// or an error.
func (r Request[T]) Do(ctx context.Context, llm *Llmberjack) (*Response[T], error) {
	if r.err != nil {
		return nil, r.err
	}

	provider, err := llm.GetProvider(r.provider)
	if err != nil {
		return nil, err
	}

	if r.ModelFunc != nil {
		if m := r.ModelFunc(provider, r.provider); m != "" {
			r.Model = &m
		}
	}

	if r.createNewThread {
		r.ThreadId = &ThreadId{
			provider: provider,
		}
	}

	if r.ThreadId != nil && r.ThreadId.provider != provider {
		return nil, errors.New("thread was not produced by provider")
	}

	resp, err := provider.ChatCompletion(ctx, llm, r)
	if err != nil {
		return nil, err
	}

	return &Response[T]{
		InnerResponse: *resp,
		ThreadId:      r.ThreadId,
	}, nil
}

func (r Request[T]) WithProvider(name string) Request[T] {
	r.provider = &name

	return r
}

func (r Request[T]) CreateThread() Request[T] {
	r.createNewThread = true

	return r
}

func (r Request[T]) InThread(threadId *ThreadId) Request[T] {
	if threadId == nil {
		r.err = errors.CombineErrors(r.err, errors.New("cannot continue a nil thread"))
		return r
	}

	r.ThreadId = threadId

	return r
}

// FromCandidate selects a candidate/choice from a previous response as the base
// for this Request.
//
// Selecting a candidate will have two effects:
//   - Adding the candidate to the history (if the request was in a thread)
//   - Using this response tool calls as a basis for tool responses, if applicable.
//
// Example usage:
//
//	resp, err := llmberjack.NewRequest[Output]().
//		FromCandidate(previousResp, 0).
//		WithText(llmberjack.RoleUser, "How are you today?").
//		Do(ctx, llm)
func (r Request[T]) FromCandidate(c Candidater, idx int) Request[T] {
	r.ThreadId = c.Thread()

	candidate, err := c.Candidate(idx)
	if err != nil {
		r.err = errors.CombineErrors(r.err, err)
		return r
	}

	r.respondsTo = candidate
	candidate.SelectCandidate()

	return r
}

func (r Request[T]) SkipSaveInput() Request[T] {
	r.innerRequest.SkipSaveInput = true

	return r
}

func (r Request[T]) SkipSaveOutput() Request[T] {
	r.innerRequest.SkipSaveOutput = true

	return r
}

// WithModel overrides the model used for this specific request.
//
// If not provided, the default model set on the provider, then the adapter will
// be used.
func (r Request[T]) WithModel(model string) Request[T] {
	r.Model = &model

	return r
}

// WithModelFunc executes a callback to determine the model to use.
//
// Is it useful notably when having multiple provider, to be able to select the
// model depending on which provider was actually selected to execute the
// request. The callback is passed the actual instance of the selected provider,
// as well as its registered name, if applicable.
func (r Request[T]) WithModelFunc(fn func(provider Llm, providerName *string) string) Request[T] {
	r.ModelFunc = fn

	return r
}

// WithInstruction adds a system prompt to the request.
//
// Note that if the adapter is configured to save history, this need only be
// added on the first request sent to the provider.
func (r Request[T]) WithInstruction(parts ...string) Request[T] {
	r.Messages = append(r.Messages, Message{
		Type: TypeText,
		Role: RoleSystem,
		Parts: lo.Map(parts, func(p string, _ int) io.Reader {
			return strings.NewReader(p)
		}),
	})

	return r
}

// WithInstructionReader adds a system prompt read from an io.Reader.
func (r Request[T]) WithInstructionReader(parts ...io.Reader) Request[T] {
	r.Messages = append(r.Messages, Message{
		Type:  TypeText,
		Role:  RoleSystem,
		Parts: parts,
	})

	return r
}

func (r Request[T]) WithInstructionFiles(files ...string) Request[T] {
	parts := make([]io.Reader, len(files))

	for idx, path := range files {
		f, err := os.Open(path)
		if err != nil {
			r.err = errors.CombineErrors(r.err, err)
			continue
		}

		parts[idx] = f
	}

	r.Messages = append(r.Messages, Message{
		Type:  TypeText,
		Role:  RoleSystem,
		Parts: parts,
	})

	return r
}

// WithText adds a text message to the Request.
//
// Each provided `string` will be added as a discrete `part` in the message. The
// message will be declared as text content.
func (r Request[T]) WithText(role MessageRole, parts ...string) Request[T] {
	r.Messages = append(r.Messages, Message{
		Type: TypeText,
		Role: role,
		Parts: lo.Map(parts, func(p string, _ int) io.Reader {
			return strings.NewReader(p)
		}),
	})

	return r
}

// WithTextReader adds a message to the Request read from an io.Reader.
func (r Request[T]) WithTextReader(role MessageRole, parts ...io.Reader) Request[T] {
	r.Messages = append(r.Messages, Message{
		Type:  TypeText,
		Role:  role,
		Parts: parts,
	})

	return r
}

func (r Request[T]) WithSerializable(role MessageRole, ser Serializer, input any) Request[T] {
	var buf bytes.Buffer

	err := ser.Serialize(input, &buf)
	if err != nil {
		r.err = errors.CombineErrors(r.err, err)
		return r
	}

	return r.WithText(role, buf.String())
}

func (r Request[T]) WithJson(role MessageRole, data any) Request[T] {
	var buf bytes.Buffer

	if err := json.NewEncoder(&buf).Encode(data); err != nil {
		r.err = errors.CombineErrors(r.err, err)
		return r
	}

	return r.WithText(role, buf.String())
}

func (r Request[T]) WithSchemaDescription(name, description string) Request[T] {
	r.SchemaName = name
	r.SchemaDescription = description

	return r
}

func (r Request[T]) OverrideResponseSchema(schema jsonschema.Schema) Request[T] {
	r.SchemaOverride = &schema

	return r
}

// WithTools adds tool definitions to the request.
//
// Tools are represented as a type-safe function taking its configuration as
// input, and return a string and an error. The JSONSchema sent to the provider
// will be generated from the input type.
//
// Example usage:
//
//	resp, err := llmberjack.NewRequest[Output]().
//		WithText(llmberjack.RoleUser, "How are you today?").
//		WithTool(llmberjack.NewTool[WeatherParams]("get_weather", "Get weather at location", llmberjack.Function(func(args WeatherParams) (string, error) {
//			return "Good weather!", nil
//		})).
//		Do(ctx, llm)
func (r Request[T]) WithTools(tools ...internal.Tool) Request[T] {
	for _, tool := range tools {
		r.Tools[tool.Name] = tool
	}

	return r
}

func (r Request[T]) withToolResponse(tool ResponseToolCall, parts string) Request[T] {
	r.Messages = append(r.Messages, Message{
		Type:  TypeText,
		Role:  RoleTool,
		Parts: []io.Reader{strings.NewReader(parts)},
		Tool:  &tool,
	})

	return r
}

// WithToolExecution executes the requested tools and add their output to the
// Request.
//
// It will also take care of adding the matching tool definitions to the
// Request, so there is not need to also call `WithTool`.
//
// Note that this requires that a candidate from the previous response was
// selected by calling `FromCandidate()` before this function, to determine
// which function the provider asked to be called.
func (r Request[T]) WithToolExecution(tools ...internal.Tool) Request[T] {
	if r.respondsTo == nil {
		r.err = errors.CombineErrors(r.err, errors.Newf("cannot execute tools without selecting a response candidate, call FromCandidate() first"))
		return r
	}
	if r.ThreadId == nil {
		r.err = errors.CombineErrors(r.err, errors.New("cannot execute tools without history, request must be in a thread"))
		return r
	}

	for _, tool := range tools {
		r = r.WithTools(tool)
	}

	for _, toolCall := range r.respondsTo.ToolCalls {
		tool, ok := r.Tools[toolCall.Name]

		if !ok {
			r.err = errors.CombineErrors(r.err, errors.Newf("no tool was registered for response to tool '%s'", toolCall.Name))
			return r
		}

		resp, err := tool.Call(toolCall.Parameters)
		if err != nil {
			r.err = errors.CombineErrors(r.err, err)
			return r
		}

		r = r.withToolResponse(toolCall, resp)
	}

	return r
}

// WithProviderOptions set provider-specific options.
//
// Some options are not going to be supported by all providers, so they will
// usually defined a type representing options specific to them. This function
// allows to define those. One set of option can be defined by provider type.
func (r Request[T]) WithProviderOptions(opts internal.ProviderRequestOptions) Request[T] {
	r.ProviderOptions[reflect.TypeOf(opts)] = opts

	return r
}

// WithMaxTokens limits how many token a provider can emit for its completion.
func (r Request[T]) WithMaxTokens(tokens int) Request[T] {
	r.MaxTokens = &tokens

	return r
}

// WithMaxCandidates limits how many candidate responses the provider is able to provide.
//
// Most providers default to 1 for this value.
func (r Request[T]) WithMaxCandidates(candidates int) Request[T] {
	r.MaxCandidates = &candidates

	return r
}

// WithTemperature sets custom temperature value to be used.
//
// Default value depends on the model.
func (r Request[T]) WithTemperature(temp float64) Request[T] {
	r.Temperature = &temp

	return r
}

// WithTopP sets the `top_p` parameter.
func (r Request[T]) WithTopP(topp float64) Request[T] {
	r.TopP = &topp

	return r
}

func (r Request[T]) WithThinking(thinking bool) Request[T] {
	r.Thinking = &thinking

	return r
}

// WithThinkingLevel sets the reasoning effort for providers that support it.
// WithThinking(false) takes precedence when both options are set.
func (r Request[T]) WithThinkingLevel(level ThinkingLevel) Request[T] {
	r.ThinkingLevel = &level

	return r
}

// WithPromptCaching enables or disables provider-managed prompt caching.
// Providers that cache prompts implicitly may ignore this setting.
func (r Request[T]) WithPromptCaching(enabled bool) Request[T] {
	r.PromptCaching = &enabled

	return r
}

// Request[T] implementation of Requester.

func (r Request[T]) ToRequest() innerRequest {
	return r.innerRequest
}

func (r Request[T]) ProviderRequestOptions(provider Llm) internal.ProviderRequestOptions {
	var providerOpts internal.ProviderRequestOptions

	if opts, ok := r.ProviderOptions[provider.RequestOptionsType()]; ok {
		providerOpts = opts
	}

	return providerOpts
}

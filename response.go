package llmberjack

import (
	"encoding/json"
	"iter"
	"time"

	"github.com/cockroachdb/errors"
)

type (
	FinishReason string
)

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonMaxTokens     FinishReason = "max_tokens"
	FinishReasonContentFilter FinishReason = "content_filter"
)

// Candidater represents a type that can have several candidates.
type Candidater interface {
	NumCandidates() int
	Candidate(int) (*ResponseCandidate, error)
	Thread() *ThreadId
}

// InnerResponse is a response from a provider.
type InnerResponse struct {
	Id         string
	Model      string
	Candidates []ResponseCandidate
	Created    time.Time
}

// ResponseCandidate represent a candidate response from a  provider.
type ResponseCandidate struct {
	Text         string
	FinishReason FinishReason
	ToolCalls    []ResponseToolCall
	Grounding    *ResponseGrounding
	Thoughts     string

	// SelectCandidate is a callback that is called when a candidate is
	// "selected" (when the conversation will continue from it).
	SelectCandidate func()
}

type ResponseGrounding struct {
	Searches []string
	Sources  []ResponseGroundingSource
	Snippets []string
}

type ResponseGroundingSource struct {
	Title  string
	Domain string
	Url    string
	Date   time.Time
}

// ResponseToolCall is a request from a provider to execute a tool.
type ResponseToolCall struct {
	Id         string
	Name       string
	Parameters []byte
}

// Response[T] is a wrapper around a provider response.
//
// It wraps it so it can be generic without the provider's response to also be,
// and provide typed methods to unmarshal the response, if necessary.
type Response[T any] struct {
	InnerResponse

	ThreadId *ThreadId
}

func (r Response[T]) NumCandidates() int {
	return len(r.Candidates)
}

func (r Response[T]) Iterator() iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		for idx := range r.Candidates {
			if !yield(r.Get(idx)) {
				return
			}
		}
	}
}

func (r Response[T]) Candidate(idx int) (*ResponseCandidate, error) {
	if idx > len(r.Candidates)-1 {
		return nil, errors.Newf("candidate %d does not exist (%d candidates)", idx, len(r.Candidates))
	}

	return &r.Candidates[idx], nil
}

func (r Response[T]) Thread() *ThreadId {
	return r.ThreadId
}

// Get will return the deserialized output for a candidate.
//
// It will parse the response and deserialize it to the requested type, or return
// an error if it cannot.
func (r Response[T]) Get(idx int) (T, error) {
	if idx > len(r.Candidates)-1 {
		return *new(T), errors.Newf("candidate %d does not exist (%d candidates)", idx, len(r.Candidates))
	}

	candidate := r.Candidates[idx]

	switch any(*new(T)).(type) {
	case string:
		return any(candidate.Text).(T), nil

	default:
		output := new(T)

		if err := json.Unmarshal([]byte(candidate.Text), output); err != nil {
			return *output, errors.Wrap(err, "failed to decode response to schema")
		}

		return *output, nil
	}
}

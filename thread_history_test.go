package llmberjack

import (
	"context"
	"testing"

	"github.com/checkmarble/llmberjack/internal"
	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// eagerFailingProvider saves the request messages to the thread history before
// failing, like a provider that does not stage its history would.
type eagerFailingProvider struct {
	*MockProvider
}

func (p eagerFailingProvider) ChatCompletion(_ context.Context, _ internal.Adapter, requester Requester) (*InnerResponse, error) {
	if req := requester.ToRequest(); req.ThreadId != nil {
		p.History.Save(req.ThreadId, MockMessage{"saved before failing"})
	}

	return nil, errors.New("provider error")
}

func TestFailedThreadCreationDoesNotLeakThread(t *testing.T) {
	provider := eagerFailingProvider{NewMockProvider()}
	provider.On("Init", mock.Anything).Return(nil)

	llm, _ := New(WithDefaultProvider(provider))

	resp, err := NewUntypedRequest().CreateThread().WithText(RoleUser, "prompt").Do(t.Context(), llm)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Len(t, provider.History.history, 0)
}

func TestFailedTurnKeepsExistingThread(t *testing.T) {
	provider := NewMockProvider()
	provider.On("Init", mock.Anything).Return(nil)

	llm, _ := New(WithDefaultProvider(provider))

	provider.On("ChatCompletion", mock.Anything, llm, mock.Anything).Return(MockMessage{"first answer"}, nil).Once()
	provider.On("ChatCompletion", mock.Anything, llm, mock.Anything).Return(nil, errors.New("provider error")).Once()

	resp, err := NewUntypedRequest().CreateThread().WithText(RoleUser, "first prompt").Do(t.Context(), llm)

	assert.NoError(t, err)

	_, err = NewUntypedRequest().InThread(resp.ThreadId).WithText(RoleUser, "second prompt").Do(t.Context(), llm)

	assert.Error(t, err)
	assert.Equal(t, []MockMessage{{"first prompt"}}, provider.History.Load(resp.ThreadId))
}

func TestSelectingCandidateTwiceSavesItOnce(t *testing.T) {
	provider := NewMockProvider()
	provider.On("Init", mock.Anything).Return(nil)

	llm, _ := New(WithDefaultProvider(provider))

	provider.On("ChatCompletion", mock.Anything, llm, mock.Anything).Return(MockMessage{"answer"}, nil).Once()

	resp, err := NewUntypedRequest().CreateThread().WithText(RoleUser, "prompt").Do(t.Context(), llm)

	assert.NoError(t, err)

	// Building a request from the same candidate twice, as when retrying a
	// failed turn, must only add the candidate to the history once.
	_ = NewUntypedRequest().FromCandidate(resp, 0)
	_ = NewUntypedRequest().FromCandidate(resp, 0)

	assert.Equal(t, []MockMessage{{"prompt"}, {"answer"}}, provider.History.Load(resp.ThreadId))
}

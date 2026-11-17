package llmberjack

import (
	"sync"
)

// History manages the conversation context by storing a sequence of messages.
// It is generic in type `T`, where `T` represents the specific message format
// required by a particular LLM provider (e.g., OpenAI's Message or AIStudio's ChatMessage).
// This allows the adapter to maintain conversation state across multiple requests.
type History[T any] struct {
	mtx     sync.Mutex
	history map[*ThreadId][]T
}

// Save appends a new message to the conversation history.
// The `message` parameter should be of the generic type `T`, matching the
// content representation expected by the LLM provider.
func (h *History[T]) Save(threadId *ThreadId, message T) {
	h.mtx.Lock()
	defer h.mtx.Unlock()

	if h.history == nil {
		h.history = make(map[*ThreadId][]T)
	}
	if _, ok := h.history[threadId]; !ok {
		h.history[threadId] = make([]T, 0)
	}

	h.history[threadId] = append(h.history[threadId], message)
}

// SaveAll atomically appends several messages to the conversation history.
//
// Providers use it to commit all the input messages of a turn at once, only
// after the provider call succeeded, so a failed call leaves the history (and
// the set of known threads) untouched. Calling it without any message is a
// no-op and does not register the thread.
func (h *History[T]) SaveAll(threadId *ThreadId, messages ...T) {
	if len(messages) == 0 {
		return
	}

	h.mtx.Lock()
	defer h.mtx.Unlock()

	if h.history == nil {
		h.history = make(map[*ThreadId][]T)
	}

	h.history[threadId] = append(h.history[threadId], messages...)
}

// Load retrieves the entire conversation history as a slice of messages.
// This history can then be included in subsequent requests to the LLM
// to maintain conversational context.
func (h *History[T]) Load(threadId *ThreadId) []T {
	h.mtx.Lock()
	defer h.mtx.Unlock()

	if h.history == nil {
		return []T{}
	}
	return h.history[threadId]
}

// Clear empties the entire conversation history, effectively starting a
// new conversation. This also clears any system instructions that were
// part of the history.
func (h *History[T]) Clear(threadId *ThreadId) {
	h.mtx.Lock()
	defer h.mtx.Unlock()

	if h.history != nil {
		h.history[threadId] = make([]T, 0)
	}
}

func (h *History[T]) Close(threadId *ThreadId) {
	h.mtx.Lock()
	defer h.mtx.Unlock()

	if h.history != nil {
		delete(h.history, threadId)
	}
}

func (h *History[T]) Copy(threadId *ThreadId) *ThreadId {
	if threadId == nil {
		return nil
	}

	h.mtx.Lock()
	defer h.mtx.Unlock()

	if _, ok := h.history[threadId]; ok {
		return nil
	}

	newThreadId := &ThreadId{provider: threadId.provider}
	newHistory := make([]T, len(h.history[threadId]))

	copy(newHistory, h.history[threadId])

	h.history[newThreadId] = newHistory

	return newThreadId
}

package llmberjack

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSaveLoadHistory(t *testing.T) {
	type message struct {
		Number int
	}

	h := History[message]{
		history: make(map[*ThreadId][]message),
	}

	assert.Len(t, h.history, 0)

	threadId := &ThreadId{}

	h.Save(threadId, message{1})
	h.Save(threadId, message{2})
	h.Save(threadId, message{3})

	assert.Len(t, h.Load(threadId), 3)
	assert.ElementsMatch(t, h.Load(threadId), []message{{1}, {2}, {3}})
}

func TestClearHistory(t *testing.T) {
	type message struct {
		Number int
	}

	h := History[message]{
		history: make(map[*ThreadId][]message),
	}

	threadId := &ThreadId{}

	assert.Len(t, h.Load(threadId), 0)

	h.Save(threadId, message{1})
	h.Save(threadId, message{2})
	h.Save(threadId, message{3})

	assert.Len(t, h.Load(threadId), 3)
	assert.ElementsMatch(t, h.Load(threadId), []message{{1}, {2}, {3}})

	h.Clear(threadId)

	assert.Len(t, h.Load(threadId), 0)
}

func TestCopyCloseThread(t *testing.T) {
	h := History[int]{
		history: make(map[*ThreadId][]int),
	}

	t1 := &ThreadId{}

	h.Save(t1, 1)
	h.Save(t1, 2)

	assert.Len(t, h.Load(t1), 2)

	t2 := h.Copy(t1)

	h.Save(t2, 3)

	assert.Len(t, h.Load(t1), 2)
	assert.Len(t, h.Load(t2), 1)
}

func TestSaveAllHistory(t *testing.T) {
	type message struct {
		Number int
	}

	h := History[message]{}
	threadId := &ThreadId{}

	h.SaveAll(threadId)

	assert.Len(t, h.history, 0, "saving no message must not register the thread")

	h.Save(threadId, message{1})
	h.SaveAll(threadId, message{2}, message{3})

	assert.Equal(t, []message{{1}, {2}, {3}}, h.Load(threadId))
}

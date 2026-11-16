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

	assert.NotNil(t, t2)
	assert.NotSame(t, t1, t2)
	assert.Equal(t, []int{1, 2}, h.Load(t2))

	h.Save(t2, 3)
	h.Save(t1, 4)

	assert.Equal(t, []int{1, 2, 4}, h.Load(t1))
	assert.Equal(t, []int{1, 2, 3}, h.Load(t2))

	h.Close(t2)

	assert.Len(t, h.Load(t2), 0)
	assert.Equal(t, []int{1, 2, 4}, h.Load(t1))
}

func TestCopyUnknownThread(t *testing.T) {
	h := History[int]{}

	assert.Nil(t, h.Copy(nil))
	assert.Nil(t, h.Copy(&ThreadId{}))

	h.Save(&ThreadId{}, 1)

	assert.Nil(t, h.Copy(&ThreadId{}))
}

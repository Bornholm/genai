package todo

// Status represents the status of a todo item
type Status string

const (
	StatusPending    Status = "pending"
	StatusInProgress Status = "in_progress"
	StatusDone       Status = "done"
)

// Item represents a single todo item
type Item struct {
	ID      string
	Content string
	Status  Status
}

// List represents a todo list
type List struct {
	Items []Item
}

// NewList creates a new empty todo list
func NewList() *List {
	return &List{
		Items: []Item{},
	}
}

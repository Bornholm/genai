package agent

type Handler interface {
	Handle(input Event, outputs chan Event) error
}

type HandlerFunc func(input Event, outputs chan Event) error

func (fn HandlerFunc) Handle(input Event, outputs chan Event) error {
	return fn(input, outputs)
}

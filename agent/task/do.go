package task

import (
	"context"

	"github.com/bornholm/genai/agent"
	"github.com/pkg/errors"
)

type DoOptions struct {
	OnThought func(evt ThoughtEvent) error
}

type DoOptionFunc func(opts *DoOptions)

func WithOnThought(onThought func(evt ThoughtEvent) error) DoOptionFunc {
	return func(opts *DoOptions) {
		opts.OnThought = onThought
	}
}

func Do(ctx context.Context, taskAgent *agent.Agent, query string, funcs ...DoOptionFunc) (ResultEvent, error) {
	opts := &DoOptions{}
	for _, fn := range funcs {
		fn(opts)
	}

	taskCtx, taskCancel := context.WithCancel(ctx)
	defer taskCancel()

	queryEvent := agent.NewMessageEvent(taskCtx, query)

	if err := taskAgent.In(queryEvent); err != nil {
		return nil, errors.WithStack(err)
	}

	for {
		select {
		case evt, ok := <-taskAgent.Output():
			if !ok {
				return nil, errors.New("output channel closed")
			}

			switch typ := evt.(type) {
			case ThoughtEvent:
				if opts.OnThought == nil || typ.Origin().ID() != queryEvent.ID() {
					continue
				}

				if err := opts.OnThought(typ); err != nil {
					return nil, errors.WithStack(err)
				}

			case ResultEvent:
				if typ.Origin().ID() != queryEvent.ID() {
					continue
				}

				return typ, nil
			}

		case err, ok := <-taskAgent.Err():
			if !ok {
				return nil, errors.New("error channel closed")
			}
			return nil, errors.WithStack(err)
		case <-ctx.Done():
			return nil, errors.WithStack(ctx.Err())
		}
	}

}

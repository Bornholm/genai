package context

import (
	"context"

	"github.com/pkg/errors"
)

var (
	ErrNotFound       = errors.New("not found")
	ErrUnexpectedType = errors.New("unexpected type")
)

type Context = context.Context

var WithValue = context.WithValue

func Value[T any, K any](ctx context.Context, key K) (T, error) {
	raw := ctx.Value(key)
	if raw == nil {
		return *new(T), errors.WithStack(ErrNotFound)
	}

	v, ok := raw.(T)
	if !ok {
		return *new(T), errors.Wrapf(ErrUnexpectedType, "unexpected value of type '%T'", raw)
	}

	return v, nil
}

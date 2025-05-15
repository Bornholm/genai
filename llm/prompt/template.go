package prompt

import (
	"bytes"
	"io"
	"io/fs"
	"text/template"

	"github.com/pkg/errors"
)

func FromFS[T any](fs fs.FS, filename string, data T, funcs ...OptionFunc) (string, error) {
	file, err := fs.Open(filename)
	if err != nil {
		return "", errors.WithStack(err)
	}

	defer file.Close()

	rawTemplate, err := io.ReadAll(file)
	if err != nil {
		return "", errors.WithStack(err)
	}

	prompt, err := Template(string(rawTemplate), data, funcs...)
	if err != nil {
		return "", errors.WithStack(err)
	}

	return prompt, nil
}

func Template[T any](rawTemplate string, data T, funcs ...OptionFunc) (string, error) {
	opts := NewOptions(funcs...)
	tmpl, err := template.New("prompt").Funcs(opts.Funcs).Parse(rawTemplate)
	if err != nil {
		return "", errors.WithStack(err)
	}

	var buff bytes.Buffer

	if err := tmpl.Execute(&buff, data); err != nil {
		return "", errors.WithStack(err)
	}

	return buff.String(), nil
}

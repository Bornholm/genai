package llm

import (
	"bytes"
	"io"
	"io/fs"
	"text/template"

	"github.com/pkg/errors"
)

func PromptTemplateFS[T any](fs fs.FS, filename string, data T) (string, error) {
	file, err := fs.Open(filename)
	if err != nil {
		return "", errors.WithStack(err)
	}

	defer file.Close()

	rawTemplate, err := io.ReadAll(file)
	if err != nil {
		return "", errors.WithStack(err)
	}

	prompt, err := PromptTemplate(string(rawTemplate), data)
	if err != nil {
		return "", errors.WithStack(err)
	}

	return prompt, nil
}

func PromptTemplate[T any](rawTemplate string, data T) (string, error) {
	tmpl, err := template.New("prompt").Parse(rawTemplate)
	if err != nil {
		return "", errors.WithStack(err)
	}

	var buff bytes.Buffer

	if err := tmpl.Execute(&buff, data); err != nil {
		return "", errors.WithStack(err)
	}

	return buff.String(), nil
}

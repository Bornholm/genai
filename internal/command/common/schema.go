package common

import (
	"encoding/json"
	"io"
	"os"

	"github.com/bornholm/genai/llm"
	"github.com/invopop/jsonschema"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"
)

func GetResponseSchema(cliCtx *cli.Context, param string) (*llm.BaseResponseSchema, error) {
	if schemaPath := cliCtx.String(param); schemaPath != "" {
		schema, err := loadJSONSchema(schemaPath)
		if err != nil {
			return nil, errors.Wrap(err, "failed to load json schema")
		}

		return llm.NewResponseSchema(
			"response",
			"Structured response according to provided schema",
			schema,
		), nil
	}

	return nil, nil
}

// loadJSONSchema loads and parses a JSON schema from a file
func loadJSONSchema(schemaPath string) (*jsonschema.Schema, error) {
	file, err := os.Open(schemaPath)
	if err != nil {
		return nil, errors.Wrap(err, "failed to open schema file")
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read schema file")
	}

	var schema jsonschema.Schema
	if err := json.Unmarshal(content, &schema); err != nil {
		return nil, errors.Wrap(err, "failed to parse JSON schema")
	}

	return &schema, nil
}

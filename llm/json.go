package llm

import (
	"encoding/json"
	"log"

	jsonrepair "github.com/RealAlexandreAI/json-repair"
	"github.com/pkg/errors"
)

func ParseJSON[T any](message Message) ([]T, error) {
	var items []T
	var parseErrors []error

	jsonBlocks := jsonBlocks(message.Content())

	for _, b := range jsonBlocks {
		var t T

		repaired, err := jsonrepair.RepairJSON(b)
		if err != nil {
			parseErrors = append(parseErrors, errors.Wrapf(err, "could not repair json: %s", b))
			log.Printf("[ERROR] %+v", errors.Wrapf(err, "could not repair json: %s", b))
			continue
		}

		if err := json.Unmarshal([]byte(repaired), &t); err != nil {
			parseErrors = append(parseErrors, errors.Wrapf(err, "invalid json: %s", b))
			log.Printf("[ERROR] %+v", errors.Wrapf(err, "invalid json: %s", b))
			continue
		}

		items = append(items, t)
	}

	// If no items were parsed and we have errors, return the errors
	if len(items) == 0 && len(parseErrors) > 0 {
		return nil, errors.Errorf("failed to parse any JSON blocks: %v", parseErrors)
	}

	return items, nil
}

// jsonBlocks returns every top-level `{…}` run of content whose braces balance,
// ignoring the braces that sit inside a JSON string. A regexp cannot do this:
// the greedy `(?mis)\{.*\}` this replaces matched from the first `{` of the
// content to its last `}`, so a stray brace anywhere in the prose around the
// answer merged everything into one unparseable block — and, since the merge
// always yielded a single match, ParseJSON never returned more than one item
// despite its slice return type.
func jsonBlocks(content string) []string {
	var blocks []string
	depth, start := 0, 0
	inString, escaped := false, false

	for i := 0; i < len(content); i++ {
		c := content[i]

		// Outside an object, quotes belong to the surrounding prose.
		if depth == 0 {
			if c == '{' {
				depth, start = 1, i
			}
			continue
		}

		if inString {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
			}
			continue
		}

		switch c {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				blocks = append(blocks, content[start:i+1])
			}
		}
	}

	return blocks
}

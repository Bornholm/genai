package llm

import (
	"encoding/json"
	"log"
	"strings"

	jsonrepair "github.com/RealAlexandreAI/json-repair"
	"github.com/pkg/errors"
)

// ParseJSON decodes into T every JSON object found in the message content, in
// the order they appear, and returns one item per object. Prose around and
// between the objects is ignored, as are stray objects: a `{}` in the prose
// decodes to a zero value and takes its place in the slice. Callers should
// therefore pick the first item carrying the field they expect rather than
// assume items[0] holds the answer.
//
// An object the model cut short is decoded too, json-repair closing it, so a
// truncated answer still yields its fields.
//
// An error is returned only when no block at all could be decoded; a block that
// fails while another succeeds is logged and skipped.
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
// ignoring the braces that sit inside a string. A regexp cannot do this: the
// greedy `(?mis)\{.*\}` this replaces matched from the first `{` of the content
// to its last `}`, so a stray brace anywhere in the prose around the answer
// merged everything into one unparseable block — and, since the merge always
// yielded a single match, ParseJSON never returned more than one item despite
// its slice return type.
//
// A brace that never closes gets one of two treatments. When what follows it
// starts like an object body, the run is a payload the model cut short, so it
// is emitted as it stands for json-repair to close. Otherwise it is prose, a
// code snippet or a template, and the scan restarts just after it so the
// objects behind it are still found.
func jsonBlocks(content string) []string {
	var blocks []string

	for {
		found, unclosed := scanJSONBlocks(content)

		blocks = append(blocks, found...)

		if unclosed < 0 {
			break
		}

		if fragment := content[unclosed:]; looksLikeObject(fragment) {
			blocks = append(blocks, fragment)
		}

		content = content[unclosed+1:]

		// Without a closing brace left, no further pass can emit anything, and
		// stopping here keeps a run of opening braces from costing a scan each.
		if strings.IndexByte(content, '}') < 0 {
			break
		}
	}

	return blocks
}

// looksLikeObject reports whether an unclosed run starts like the body of a JSON
// object rather than like prose: a quote is a key about to be written, anything
// else is a brace the model used for something other than JSON.
func looksLikeObject(fragment string) bool {
	for i := 1; i < len(fragment); i++ {
		switch fragment[i] {
		case ' ', '\t', '\r', '\n':
			continue
		case '"', '\'':
			return true
		default:
			return false
		}
	}

	return false
}

// scanJSONBlocks scans content once and returns the balanced top-level blocks it
// contains, plus the index of the opening brace left unclosed at the end of the
// content, or -1 when every brace balanced.
func scanJSONBlocks(content string) ([]string, int) {
	var blocks []string
	depth, start := 0, 0
	var quote byte
	escaped := false

	for i := 0; i < len(content); i++ {
		c := content[i]

		// Outside an object, quotes belong to the surrounding prose.
		if depth == 0 {
			if c == '{' {
				depth, start = 1, i
			}
			continue
		}

		if quote != 0 {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == quote:
				quote = 0
			}
			continue
		}

		switch c {
		// json-repair accepts single quoted strings, so the scan has to follow
		// them too: a `}` inside one does not close the object.
		case '"', '\'':
			quote = c
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				blocks = append(blocks, content[start:i+1])
			}
		}
	}

	if depth > 0 {
		return blocks, start
	}

	return blocks, -1
}

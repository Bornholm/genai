package llm

import (
	"encoding/json"
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
// An error is returned only when no block at all could be decoded. A block that
// fails while another succeeds is skipped silently, so a stray `{}` decoding to
// a zero value is enough to hide the failure of the block that carried the
// answer. Callers that need to tell "nothing found" from "the answer was
// unreadable" should check the fields they expect, not just the error.
//
// A run cut short spans the objects that closed inside it, so those objects are
// returned twice: once on their own and once inside the repaired run. Callers
// that aggregate over every item should deduplicate.
//
// A run left open whose body starts with an unquoted key, as in `{category:
// "doc"`, is read as prose and dropped, even though json-repair would accept it
// closed. Telling it apart from a brace used for something other than JSON is
// not worth the junk items the looser test would let through.
func ParseJSON[T any](message Message) ([]T, error) {
	var items []T
	var parseErrors []error

	jsonBlocks := jsonBlocks(message.Content())

	for _, b := range jsonBlocks {
		var t T

		// Splitting the content into several blocks makes undecodable ones
		// routine: prose braces, truncated runs and the objects nested in them.
		// They are collected in parseErrors rather than logged, so an answer
		// that parses does not print [ERROR] from inside a library.
		repaired, err := jsonrepair.RepairJSON(b)
		if err != nil {
			parseErrors = append(parseErrors, errors.Wrapf(err, "could not repair json: %s", excerpt(b)))
			continue
		}

		if err := json.Unmarshal([]byte(repaired), &t); err != nil {
			parseErrors = append(parseErrors, errors.Wrapf(err, "invalid json: %s", excerpt(b)))
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

// excerpt shortens a block for an error message. A run the model cut short
// covers the rest of its answer, so quoting it whole means a caller logging the
// error prints the whole model output.
func excerpt(block string) string {
	const max = 200

	if len(block) <= max {
		return block
	}

	return block[:max] + "…"
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
// is emitted as it stands for json-repair to close, after the blocks that
// closed on their own. Otherwise it is prose, a code snippet or a template, and
// the scan restarts just after it so the objects behind it are still found.
func jsonBlocks(content string) []string {
	var blocks []string

	var truncated string

	for {
		found, unclosed := scanJSONBlocks(content)

		blocks = append(blocks, found...)

		if unclosed < 0 {
			break
		}

		// The outermost run is the payload the model was cutting short; the ones
		// the restart walks into sit inside prose it already skipped.
		if truncated == "" && looksLikeObject(content[unclosed:]) {
			truncated = content[unclosed:]
		}

		// Nothing between here and the next brace can open a block, so the scan
		// moves brace to brace rather than a byte at a time.
		next := strings.IndexByte(content[unclosed+1:], '{')
		if next < 0 {
			break
		}

		content = content[unclosed+1+next:]

		if strings.IndexByte(content, '}') >= 0 {
			continue
		}

		// No brace can close any more, so no further pass can emit a balanced
		// block. A payload the model cut short can still be in there, and one
		// walk finds it.
		if truncated == "" {
			for i := 0; i < len(content); i++ {
				if content[i] == '{' && looksLikeObject(content[i:]) {
					truncated = content[i:]
					break
				}
			}
		}

		break
	}

	// The truncated run spans everything the restart found inside it, so it goes
	// last: a caller looking for the first item carrying its field should meet
	// the objects that closed on their own before the one json-repair guessed at.
	if truncated != "" {
		blocks = append(blocks, truncated)
	}

	return blocks
}

// looksLikeObject reports whether an unclosed run starts like the body of a JSON
// object rather than like prose. A quoted key followed by a colon is a model
// writing an object; a lone quote is not enough, since prose quotes a brace
// often enough (`use "{" as delimiter`) to turn into a junk item.
func looksLikeObject(fragment string) bool {
	i := skipSpace(fragment, 1)
	if i >= len(fragment) {
		return false
	}

	quote := fragment[i]
	if quote != '"' && quote != '\'' {
		return false
	}

	for i++; i < len(fragment); i++ {
		if fragment[i] == '\\' {
			i++
			continue
		}

		if fragment[i] == quote {
			break
		}
	}

	// A key the content never closes is prose, not a key.
	if i >= len(fragment) {
		return false
	}

	i = skipSpace(fragment, i+1)

	return i < len(fragment) && fragment[i] == ':'
}

func skipSpace(s string, i int) int {
	for ; i < len(s); i++ {
		switch s[i] {
		case ' ', '\t', '\r', '\n':
		default:
			return i
		}
	}

	return i
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

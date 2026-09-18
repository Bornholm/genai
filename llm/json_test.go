package llm

import "testing"

type testVerdict struct {
	Category   string  `json:"category"`
	Confidence float64 `json:"confidence"`
}

func TestParseJSON(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
		want    []string
	}{
		{
			name:    "bare object",
			content: `{"category":"doc"}`,
			want:    []string{"doc"},
		},
		{
			name:    "object wrapped in prose",
			content: "Sure thing! Here you go:\n{\"category\": \"doc\"}",
			want:    []string{"doc"},
		},
		{
			name:    "object in a markdown fence",
			content: "```json\n{\"category\": \"doc\"}\n```",
			want:    []string{"doc"},
		},
		{
			// The greedy regexp merged the stray pair and the answer into one
			// block, which decoded to nothing.
			name:    "stray braces before the answer",
			content: `the model hesitated {} then answered {"category":"doc"}`,
			want:    []string{"", "doc"},
		},
		{
			name:    "stray object before the answer",
			content: `scratchpad {"a":1} answer {"category":"doc"}`,
			want:    []string{"", "doc"},
		},
		{
			// The slice return type promises this; the greedy regexp never
			// delivered more than one item.
			name:    "several objects",
			content: `{"category":"doc"} and {"category":"code"}`,
			want:    []string{"doc", "code"},
		},
		{
			name:    "braces inside a string value",
			content: `{"category":"doc","reason":"about func(){} syntax"}`,
			want:    []string{"doc"},
		},
		{
			name:    "escaped quote inside a string value",
			content: `{"category":"doc","reason":"he said \"} \" and left"}`,
			want:    []string{"doc"},
		},
		{
			name:    "nested object",
			content: `{"category":"doc","meta":{"tokens":12}}`,
			want:    []string{"doc"},
		},
		{
			name:    "trailing comma is repaired",
			content: `{"category":"doc",}`,
			want:    []string{"doc"},
		},
		{
			// An unpaired brace in the prose used to leave the scan one level
			// deep for the rest of the content, so the answer was never emitted.
			name:    "orphan opening brace before the answer",
			content: `the model wrote { then answered {"category":"doc"}`,
			want:    []string{"doc"},
		},
		{
			// The orphan brace starts like an object body, so it is handed to
			// json-repair as a truncated payload; the answer behind it is still
			// recovered by the restart.
			name:    "orphan opening brace then unpaired quote",
			content: `oops {" then {"category":"doc"}`,
			want:    []string{"", "doc"},
		},
		{
			name:    "quoted braces in the prose",
			content: `use format "{x}" then {"category":"doc"}`,
			want:    []string{"", "doc"},
		},
		{
			name:    "quotes in the prose around the answer",
			content: `He said "here" {"category":"doc"}`,
			want:    []string{"doc"},
		},
		{
			// json-repair accepts single quoted strings, so a `}` inside one
			// must not close the object early.
			name:    "closing brace inside a single quoted value",
			content: `{'category':'doc','reason':'a}b'}`,
			want:    []string{"doc"},
		},
		{
			name:    "no object at all",
			content: "I think this is documentation.",
			want:    nil,
		},
		{
			// Cut off by a token limit: json-repair closes the object, so the
			// answer survives.
			name:    "unterminated object",
			content: `{"category":"doc"`,
			want:    []string{"doc"},
		},
		{
			// The nested object closes, so the scan never returns to depth 0. The
			// repaired outer object now comes first; `meta` follows as an object
			// found in the content, where it used to stand in for the answer.
			name:    "truncated object with a closed nested object",
			content: `{"category":"doc","meta":{"tokens":12}`,
			want:    []string{"doc", ""},
		},
		{
			// An unescaped apostrophe flips the single quote parity, so the
			// closing brace is read as string content and the object never
			// closes. json-repair handles the fragment.
			name:    "apostrophe inside a single quoted value",
			content: `{'reason': 'it's ok', 'category': 'doc'}`,
			want:    []string{"doc"},
		},
		{
			name:    "two unbalanced runs before the answer",
			content: `a { b { {"category":"doc"}`,
			want:    []string{"doc"},
		},
		{
			name:    "answer then an orphan brace",
			content: `{"category":"doc"} oops {"bad`,
			want:    []string{"doc", ""},
		},
		{
			// A brace the model did not use for JSON must not become an item.
			name:    "brace in the prose that opens nothing",
			content: "use { for blocks",
			want:    nil,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			items, err := ParseJSON[testVerdict](NewMessage(RoleAssistant, tc.content))
			if err != nil {
				t.Fatalf("unexpected error: %+v", err)
			}
			if len(items) != len(tc.want) {
				t.Fatalf("expected %d item(s) %v, got %d: %+v", len(tc.want), tc.want, len(items), items)
			}
			for i, want := range tc.want {
				if items[i].Category != want {
					t.Errorf("item %d: expected category %q, got %q", i, want, items[i].Category)
				}
			}
		})
	}
}

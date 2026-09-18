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
		want    []testVerdict
		wantErr bool
	}{
		{
			name:    "bare object",
			content: `{"category":"doc"}`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			name:    "object wrapped in prose",
			content: "Sure thing! Here you go:\n{\"category\": \"doc\"}",
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			name:    "object in a markdown fence",
			content: "```json\n{\"category\": \"doc\"}\n```",
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			// The greedy regexp merged the stray pair and the answer into one
			// block, which decoded to nothing.
			name:    "stray braces before the answer",
			content: `the model hesitated {} then answered {"category":"doc"}`,
			want:    []testVerdict{{Category: ""}, {Category: "doc"}},
		},
		{
			name:    "stray object before the answer",
			content: `scratchpad {"a":1} answer {"category":"doc"}`,
			want:    []testVerdict{{Category: ""}, {Category: "doc"}},
		},
		{
			// The slice return type promises this; the greedy regexp never
			// delivered more than one item.
			name:    "several objects",
			content: `{"category":"doc"} and {"category":"code"}`,
			want:    []testVerdict{{Category: "doc"}, {Category: "code"}},
		},
		{
			name:    "braces inside a string value",
			content: `{"category":"doc","reason":"about func(){} syntax"}`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			name:    "escaped quote inside a string value",
			content: `{"category":"doc","reason":"he said \"} \" and left"}`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			name:    "nested object",
			content: `{"category":"doc","meta":{"tokens":12}}`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			name:    "trailing comma is repaired",
			content: `{"category":"doc",}`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			// An unpaired brace in the prose used to leave the scan one level
			// deep for the rest of the content, so the answer was never emitted.
			name:    "orphan opening brace before the answer",
			content: `the model wrote { then answered {"category":"doc"}`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			// The orphan brace starts like an object body, so it is handed to
			// json-repair as a truncated payload. It lands behind the answer the
			// restart recovers, which json-repair cannot make sense of.
			name:    "orphan opening brace then unpaired quote",
			content: `oops {" then {"category":"doc"}`,
			want:    []testVerdict{{Category: "doc"}, {Category: ""}},
		},
		{
			name:    "quoted braces in the prose",
			content: `use format "{x}" then {"category":"doc"}`,
			want:    []testVerdict{{Category: ""}, {Category: "doc"}},
		},
		{
			name:    "quotes in the prose around the answer",
			content: `He said "here" {"category":"doc"}`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			// json-repair accepts single quoted strings, so a `}` inside one
			// must not close the object early.
			name:    "closing brace inside a single quoted value",
			content: `{'category':'doc','reason':'a}b'}`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			name:    "every field is decoded",
			content: `{"category":"doc","confidence":0.9}`,
			want:    []testVerdict{{Category: "doc", Confidence: 0.9}},
		},
		{
			// Pins json-repair, not intended behaviour: reserialising a fragment
			// sends its numbers through float32, so 0.9 comes back as
			// 0.8999999761581421. Callers comparing floats from a truncated
			// answer need a tolerance.
			name:    "truncated payload loses float precision",
			content: `{"category":"doc","confidence":0.9`,
			want:    []testVerdict{{Category: "doc", Confidence: 0.8999999761581421}},
		},
		{
			// The prose brace leaves no closing brace anywhere, which used to
			// stop the scan before the cut-off answer behind it was qualified.
			name:    "prose brace before a truncated answer",
			content: `use { for blocks. Answer: {"category":"doc"`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			name:    "unpaired quote before a truncated answer",
			content: `say "{oops then {"category":"doc`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			// A category that is an object cannot decode into a string.
			name:    "every block fails to decode",
			content: `{"category":{"a":1}}`,
			wantErr: true,
		},
		{
			// The stray `{}` decodes, so the failure of the block carrying the
			// answer is swallowed and the caller gets a zero value, no error.
			// Documented in the godoc; pinned here so it stays deliberate.
			name:    "a stray object hides the failure of the real one",
			content: `{} and {"category":{"a":1}}`,
			want:    []testVerdict{{}},
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
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			// The nested object closes, so the scan never returns to depth 0. It
			// used to stand in for the answer; the repaired outer object now
			// follows it and carries the category a caller looks for.
			name:    "truncated object with a closed nested object",
			content: `{"category":"doc","meta":{"tokens":12}`,
			want:    []testVerdict{{Category: ""}, {Category: "doc"}},
		},
		{
			// An unescaped apostrophe flips the single quote parity, so the
			// closing brace is read as string content and the object never
			// closes. json-repair handles the fragment.
			name:    "apostrophe inside a single quoted value",
			content: `{'reason': 'it's ok', 'category': 'doc'}`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			// The draft spans the answer, so json-repair sees both. The answer
			// closed on its own, so it comes first whatever json-repair makes of
			// the draft.
			name:    "draft object left open before the answer",
			content: `scratchpad {"category":"code" then the answer {"category":"doc"}`,
			want:    []testVerdict{{Category: "doc"}, {Category: "doc"}},
		},
		{
			name:    "two unbalanced runs before the answer",
			content: `a { b { {"category":"doc"}`,
			want:    []testVerdict{{Category: "doc"}},
		},
		{
			name:    "answer then an orphan brace",
			content: `{"category":"doc"} oops {"bad`,
			want:    []testVerdict{{Category: "doc"}, {Category: ""}},
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
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected an error, got items: %+v", items)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %+v", err)
			}
			if len(items) != len(tc.want) {
				t.Fatalf("expected %d item(s) %+v, got %d: %+v", len(tc.want), tc.want, len(items), items)
			}
			for i, want := range tc.want {
				if items[i] != want {
					t.Errorf("item %d: expected %+v, got %+v", i, want, items[i])
				}
			}
		})
	}
}

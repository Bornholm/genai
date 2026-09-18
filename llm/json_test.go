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
			name:    "no object at all",
			content: "I think this is documentation.",
			want:    nil,
		},
		{
			name:    "unterminated object",
			content: `{"category":"doc"`,
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

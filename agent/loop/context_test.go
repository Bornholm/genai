package loop

import (
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

// buildToolGroup creates a [ToolCallsMessage, ToolMessage] pair with the given result content.
func buildToolGroup(id, toolName, result string) []llm.Message {
	tc := llm.NewToolCall(id, toolName, `{}`)
	return []llm.Message{
		llm.NewToolCallsMessage(tc),
		llm.NewToolMessage(id, llm.NewToolResult(result)),
	}
}

// countTokens uses the default estimator (len/4).
func countTokens(s string) int { return defaultTokenEstimator(s) }

func TestCompressingStrategy_UnderLimit(t *testing.T) {
	strategy := DefaultCompressingTruncationStrategy(10000, countTokens, DefaultCompressionRatio)

	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "system"),
		llm.NewMessage(llm.RoleUser, "hello"),
	}

	got, err := strategy(messages)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(messages) {
		t.Fatalf("expected %d messages, got %d", len(messages), len(got))
	}
}

func TestCompressingStrategy_CompressesOldToolResults(t *testing.T) {
	// Budget that can hold 3 compressed old results + 3 large recent results,
	// but not 5 full large results.
	const maxTokens = 5000

	strategy := DefaultCompressingTruncationStrategy(maxTokens, countTokens, DefaultCompressionRatio)

	// Build a large tool result (definitely over the compressed threshold).
	largeResult := strings.Repeat("x", 5000) // 1250 tokens each

	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "sys"),
		llm.NewMessage(llm.RoleUser, "user"),
	}
	// 5 groups: the last 3 are protected from compression (only groups 0 & 1 can be compressed).
	for i := range 5 {
		id := string(rune('a' + i))
		messages = append(messages, buildToolGroup(id, "search", largeResult)...)
	}

	initialTokens := estimateMessagesTokens(messages, countTokens)
	got, err := strategy(messages)
	if err != nil {
		t.Fatal(err)
	}

	// Verify that tokens were reduced and we are within budget.
	total := estimateMessagesTokens(got, countTokens)
	if total >= initialTokens {
		t.Errorf("expected compression to reduce total tokens (was %d, got %d)", initialTokens, total)
	}
	if total > maxTokens {
		t.Errorf("expected result under %d tokens, got %d", maxTokens, total)
	}

	// Verify the 3 most recent tool results are preserved at full size.
	toolMsgs := []llm.Message{}
	for _, m := range got {
		if m.Role() == llm.RoleTool {
			toolMsgs = append(toolMsgs, m)
		}
	}
	if len(toolMsgs) < 3 {
		t.Fatalf("expected at least 3 tool messages, got %d", len(toolMsgs))
	}
	for _, m := range toolMsgs[len(toolMsgs)-3:] {
		if m.Content() != largeResult {
			t.Error("most recent tool results should be preserved unmodified")
		}
	}
}

func TestCompressingStrategy_FallsBackToDropping(t *testing.T) {
	// Extremely tight budget: forces both compression AND dropping of old groups.
	const maxTokens = 100

	strategy := DefaultCompressingTruncationStrategy(maxTokens, countTokens, DefaultCompressionRatio)

	largeResult := strings.Repeat("y", 2000)

	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "sys"),
		llm.NewMessage(llm.RoleUser, "user"),
	}
	for i := range 6 {
		id := string(rune('a' + i))
		messages = append(messages, buildToolGroup(id, "tool", largeResult)...)
	}

	got, err := strategy(messages)
	if err != nil {
		t.Fatal(err)
	}

	// Result must always contain the two anchor messages.
	if got[0].Role() != llm.RoleSystem {
		t.Error("first message should always be the system anchor")
	}
	if got[1].Role() != llm.RoleUser {
		t.Error("second message should always be the user anchor")
	}
}

func TestCompressingStrategy_PreservesRecentGroupsIntact(t *testing.T) {
	const maxTokens = 300

	strategy := DefaultCompressingTruncationStrategy(maxTokens, countTokens, DefaultCompressionRatio)

	largeResult := strings.Repeat("z", 3000)
	recentResult := "short recent result"

	messages := []llm.Message{
		llm.NewMessage(llm.RoleSystem, "sys"),
		llm.NewMessage(llm.RoleUser, "user"),
	}
	// 2 old groups with large results + 3 recent groups with a known short result.
	for i := range 2 {
		id := string(rune('a' + i))
		messages = append(messages, buildToolGroup(id, "tool", largeResult)...)
	}
	for i := range 3 {
		id := string(rune('c' + i))
		messages = append(messages, buildToolGroup(id, "tool", recentResult)...)
	}

	got, err := strategy(messages)
	if err != nil {
		t.Fatal(err)
	}

	// All tool messages in the output from the 3 recent groups should be unmodified.
	for _, m := range got {
		if m.Role() != llm.RoleTool {
			continue
		}
		if m.Content() == recentResult {
			continue // correct
		}
		// If it's a compressed/truncated version, it must NOT be from a recent group.
		if strings.HasSuffix(m.Content(), "]") {
			// Check it's not the recent short result.
			if strings.Contains(m.Content(), recentResult) {
				t.Error("recent tool result should not have been modified")
			}
		}
	}
}

func TestCompressGroupToolResults_TruncatesLargeContent(t *testing.T) {
	const maxTokens = 10

	largeContent := strings.Repeat("a", 1000)
	tc := llm.NewToolCall("id1", "fn", `{}`)
	group := messageGroup{
		llm.NewToolCallsMessage(tc),
		llm.NewToolMessage("id1", llm.NewToolResult(largeContent)),
	}

	compressed, saved := compressGroupToolResults(group, maxTokens, countTokens)

	if saved <= 0 {
		t.Errorf("expected tokens to be saved, got %d", saved)
	}
	toolMsg := compressed[1]
	if len(toolMsg.Content()) >= len(largeContent) {
		t.Error("content should have been truncated")
	}
	if !strings.Contains(toolMsg.Content(), "[Result compressed:") {
		t.Error("truncated content should include compression notice")
	}
}

func TestCompressGroupToolResults_LeavesSmallContentUntouched(t *testing.T) {
	const maxTokens = 1000

	smallContent := "small result"
	tc := llm.NewToolCall("id1", "fn", `{}`)
	group := messageGroup{
		llm.NewToolCallsMessage(tc),
		llm.NewToolMessage("id1", llm.NewToolResult(smallContent)),
	}

	compressed, saved := compressGroupToolResults(group, maxTokens, countTokens)

	if saved != 0 {
		t.Errorf("expected 0 tokens saved for small content, got %d", saved)
	}
	if compressed[1].Content() != smallContent {
		t.Error("small content should be unchanged")
	}
}

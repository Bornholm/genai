package conformance

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/bornholm/genai/llm"
)

// cacheablePrefix builds a system prompt long enough to be cached by every
// provider: Anthropic requires at least 4096 tokens on Haiku 4.5, the
// strictest published minimum. Two hundred lines of this prose, about
// 42 000 characters, land well above it whatever the tokenizer, and the
// prefix is sent four times per run so it is kept no longer than that.
func cacheablePrefix() string {
	var b strings.Builder
	b.WriteString("You are a meticulous archivist. The following reference material describes an imaginary city; answer questions about it briefly.\n\n")
	for i := 0; i < 200; i++ {
		fmt.Fprintf(&b, "District %d is bounded by canal %d to the north and by the old rampart to the south; its market opens on day %d of each week, its guild hall was rebuilt in year %d after the great flood, and its census counts %d households.\n",
			i, i%17, i%7+1, 1200+i, 300+i*7)
	}
	return b.String()
}

// testCaching checks that an explicit cache hint on a long prefix is
// honoured by the provider: the second call with the same prefix must be
// served from the cache, which the usage reports as cached tokens.
func testCaching(t *testing.T, client any) {
	t.Helper()

	chatClient, ok := client.(llm.ChatCompletionClient)
	if !ok {
		t.Skip("client does not implement ChatCompletionClient")
	}

	ctx := context.Background()
	prefix := cacheablePrefix()

	ask := func(t *testing.T, prefix string, cc *llm.CacheControl, question string) llm.ChatCompletionUsage {
		t.Helper()
		res, err := chatClient.ChatCompletion(ctx,
			llm.WithMessages(
				llm.NewMessageWithCacheControl(llm.RoleSystem, prefix, cc),
				llm.NewMessage(llm.RoleUser, question),
			),
			llm.WithTemperature(0),
			llm.WithMaxCompletionTokens(64),
		)
		if err != nil {
			t.Fatalf("ChatCompletion error: %v", err)
		}
		if res.Usage() == nil {
			t.Fatal("expected non-nil usage")
		}
		return res.Usage()
	}

	type cachedUsage interface{ CachedTokens() int64 }

	t.Run("PrefixCacheHit", func(t *testing.T) {
		cc := &llm.CacheControl{Type: "ephemeral"}

		first := ask(t, prefix, cc, "Which canal bounds district 3? Answer with the number only.")
		if cw, ok := first.(llm.CacheCreationReportingUsage); ok && cw.CacheCreationTokens() == 0 {
			if cu, ok := first.(cachedUsage); !ok || cu.CachedTokens() == 0 {
				t.Logf("first call reports neither cache writes nor cache reads; the prefix may already be cached from a previous run")
			}
		}

		second := ask(t, prefix, cc, "Which canal bounds district 5? Answer with the number only.")

		// Raw counters, kept in the log as evidence of the usage
		// decomposition the provider applies.
		for label, usage := range map[string]llm.ChatCompletionUsage{"cold": first, "warm": second} {
			var read, written int64
			if cu, ok := usage.(cachedUsage); ok {
				read = cu.CachedTokens()
			}
			if cw, ok := usage.(llm.CacheCreationReportingUsage); ok {
				written = cw.CacheCreationTokens()
			}
			t.Logf("%s call: prompt=%d cache_read=%d cache_creation=%d completion=%d",
				label, usage.PromptTokens(), read, written, usage.CompletionTokens())
		}

		// Both calls carry the same prompt: their prompt token counts must
		// match closely whether the prefix was written or read. A count
		// twice as large on the writing call would mean the provider adds
		// cache writes to an input count that already includes them.
		if lo, hi := first.PromptTokens(), second.PromptTokens(); lo > hi*11/10 || hi > lo*11/10 {
			t.Errorf("prompt tokens differ between the cache write (%d) and the cache read (%d) of the same prompt: the usage decomposition double counts", lo, hi)
		}

		cu, ok := second.(cachedUsage)
		if !ok {
			t.Fatal("usage does not report cached tokens")
		}
		if cu.CachedTokens() == 0 {
			t.Errorf("expected the second call to be served from the cache, got 0 cached tokens (prompt=%d)", second.PromptTokens())
		}
		if cu.CachedTokens() > second.PromptTokens() {
			t.Errorf("cached tokens (%d) exceed prompt tokens (%d): the usage decomposition is wrong", cu.CachedTokens(), second.PromptTokens())
		}
	})

	t.Run("ExtendedTTL", func(t *testing.T) {
		ttl := "1h"
		cc := &llm.CacheControl{Type: "ephemeral", TTL: &ttl}

		// The prefix differs from the previous subtest so the 1h entry is
		// written here, not read from the 5m one.
		extended := "Extended cache variant.\n" + prefix
		ask(t, extended, cc, "Which canal bounds district 7? Answer with the number only.")
		second := ask(t, extended, cc, "Which canal bounds district 9? Answer with the number only.")
		if cu, ok := second.(cachedUsage); !ok || cu.CachedTokens() == 0 {
			t.Errorf("expected a cache hit on a prefix annotated with a 1h TTL, got %v", second)
		}
	})
}

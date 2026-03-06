package filter

import (
	"context"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/bornholm/genai/llm"
	"github.com/bornholm/genai/proxy"
	"github.com/pkg/errors"
)

// ContentFilter is a PreRequestHook that applies a set of FilterRules to
// the messages in a chat completion request.
type ContentFilter struct {
	rules    []FilterRule
	priority int
}

// Name implements proxy.Hook.
func (f *ContentFilter) Name() string { return "filter.content" }

// Priority implements proxy.Hook.
func (f *ContentFilter) Priority() int { return f.priority }

// PreRequest implements proxy.PreRequestHook.
func (f *ContentFilter) PreRequest(ctx context.Context, req *proxy.ProxyRequest) (*proxy.HookResult, error) {
	if req.Type != proxy.RequestTypeChatCompletion {
		return nil, nil
	}

	// Extract messages from ChatOptions
	messages := extractMessages(req.ChatOptions)

	for _, rule := range f.rules {
		if err := rule.Check(ctx, messages); err != nil {
			apiErr := proxy.NewBadRequestError(fmt.Sprintf("content policy violation: %s", err.Error()))
			return &proxy.HookResult{
				Response: &proxy.ProxyResponse{
					StatusCode: http.StatusBadRequest,
					Body:       proxy.ErrorResponse{Error: *apiErr},
				},
			}, nil
		}
	}

	return nil, nil
}

// extractMessages reconstructs the message list from ChatCompletionOptionFuncs.
func extractMessages(funcs []llm.ChatCompletionOptionFunc) []llm.Message {
	opts := llm.NewChatCompletionOptions(funcs...)
	return opts.Messages
}

// NewContentFilter creates a ContentFilter with the given rules.
func NewContentFilter(priority int, rules ...FilterRule) *ContentFilter {
	return &ContentFilter{rules: rules, priority: priority}
}

var _ proxy.PreRequestHook = &ContentFilter{}

// ---- Built-in rules -----------------------------------------------------

// RegexRule blocks messages matching a regular expression.
type RegexRule struct {
	pattern *regexp.Regexp
}

func (r *RegexRule) Check(ctx context.Context, messages []llm.Message) error {
	for _, msg := range messages {
		if r.pattern.MatchString(msg.Content()) {
			return errors.Errorf("message matches blocked pattern %q", r.pattern.String())
		}
	}
	return nil
}

// NewRegexRule compiles pattern and returns a RegexRule.
func NewRegexRule(pattern string) (*RegexRule, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, errors.Wrapf(err, "invalid regex pattern %q", pattern)
	}
	return &RegexRule{pattern: re}, nil
}

var _ FilterRule = &RegexRule{}

// KeywordRule blocks messages containing any of the listed keywords (case-insensitive).
type KeywordRule struct {
	keywords []string
}

func (k *KeywordRule) Check(ctx context.Context, messages []llm.Message) error {
	for _, msg := range messages {
		lower := strings.ToLower(msg.Content())
		for _, kw := range k.keywords {
			if strings.Contains(lower, strings.ToLower(kw)) {
				return errors.Errorf("message contains blocked keyword %q", kw)
			}
		}
	}
	return nil
}

// NewKeywordRule creates a KeywordRule for the given keywords.
func NewKeywordRule(keywords ...string) *KeywordRule {
	return &KeywordRule{keywords: keywords}
}

var _ FilterRule = &KeywordRule{}

// MaxTokenRule blocks requests whose total estimated character count exceeds a limit.
// (Uses character count as a proxy for token count; not exact but avoids dependencies.)
type MaxTokenRule struct {
	maxChars int
}

func (m *MaxTokenRule) Check(ctx context.Context, messages []llm.Message) error {
	total := 0
	for _, msg := range messages {
		total += len(msg.Content())
	}
	if total > m.maxChars {
		return errors.Errorf("total message size %d exceeds limit %d", total, m.maxChars)
	}
	return nil
}

// NewMaxTokenRule creates a rule that limits total message content size (in characters).
func NewMaxTokenRule(maxChars int) *MaxTokenRule {
	return &MaxTokenRule{maxChars: maxChars}
}

var _ FilterRule = &MaxTokenRule{}

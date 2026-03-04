package agent

import (
	"fmt"
	"strings"
	"time"

	"charm.land/lipgloss/v2"

	"github.com/bornholm/genai/agent"
)

var (
	// Color palette
	primaryColor   = lipgloss.Color("86")
	secondaryColor = lipgloss.Color("75")
	successColor   = lipgloss.Color("76")
	warningColor   = lipgloss.Color("226")
	errorColor     = lipgloss.Color("196")
	infoColor      = lipgloss.Color("39")
	mutedColor     = lipgloss.Color("242")
	reasoningColor = lipgloss.Color("219")
	timeColor      = lipgloss.Color("245")

	// Text styles
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(primaryColor).
			Padding(0, 0, 1, 0)

	subtleStyle = lipgloss.NewStyle().
			Foreground(mutedColor).
			Italic(true)

	successStyle = lipgloss.NewStyle().
			Foreground(successColor).
			Bold(true)

	errorStyle = lipgloss.NewStyle().
			Foreground(errorColor).
			Bold(true)

	warningStyle = lipgloss.NewStyle().
			Foreground(warningColor).
			Bold(true)

	infoStyle = lipgloss.NewStyle().
			Foreground(infoColor)

	reasoningStyle = lipgloss.NewStyle().
			Foreground(reasoningColor).
			Italic(true)

	toolNameStyle = lipgloss.NewStyle().
			Foreground(secondaryColor).
			Bold(true)

	toolResultStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("252"))

	timeStyle = lipgloss.NewStyle().
			Foreground(timeColor).
			Italic(true)
)

// RenderEvent renders an agent event with lipgloss styling
func RenderEvent(evt agent.Event) string {
	switch evt.Type() {
	case agent.EventTypeComplete:
		data := evt.Data().(*agent.CompleteData)
		return renderComplete(data)
	case agent.EventTypeToolCallStart:
		data := evt.Data().(*agent.ToolCallStartData)
		return renderToolCallStart(data)
	case agent.EventTypeToolCallDone:
		data := evt.Data().(*agent.ToolCallDoneData)
		return renderToolCallDone(data)
	case agent.EventTypeTodoUpdated:
		data := evt.Data().(*agent.TodoUpdatedData)
		return renderTodoUpdated(data)
	case agent.EventTypeReasoning:
		data := evt.Data().(*agent.ReasoningData)
		return renderReasoning(data)
	case agent.EventTypeError:
		data := evt.Data().(*agent.ErrorData)
		return renderError(data)
	default:
		return ""
	}
}

func renderComplete(data *agent.CompleteData) string {
	if data.Message == "" {
		return ""
	}

	timestamp := timeStyle.Render(formatTime(time.Now()))
	header := titleStyle.Render("✓ Task Complete")
	divider := subtleStyle.Render(strings.Repeat("─", 60))

	return fmt.Sprintf("%s %s\n%s\n\n%s\n", timestamp, header, divider, data.Message)
}

func renderToolCallStart(data *agent.ToolCallStartData) string {
	timestamp := timeStyle.Render(formatTime(time.Now()))
	icon := "⚡"
	header := fmt.Sprintf("%s %s", icon, toolNameStyle.Render("Tool: "+data.Name))

	// Format parameters nicely
	var paramsStr string
	if data.Parameters != nil {
		paramsStr = fmt.Sprintf("%v", data.Parameters)
		// Truncate long parameter output
		if len(paramsStr) > 200 {
			paramsStr = paramsStr[:200] + "..."
		}
	}

	return fmt.Sprintf("\n%s %s\n%s\n", timestamp, header, paramsStr)
}

func renderToolCallDone(data *agent.ToolCallDoneData) string {
	timestamp := timeStyle.Render(formatTime(time.Now()))
	icon := "✓"
	header := fmt.Sprintf("%s %s", successStyle.Render(icon), toolNameStyle.Render("Tool: "+data.Name))

	// Truncate long result output
	result := data.Result
	if len(result) > 500 {
		result = result[:500] + "\n... (truncated)"
	}

	return fmt.Sprintf("\n%s %s\n%s\n", timestamp, header, toolResultStyle.Render(result))
}

func renderTodoUpdated(data *agent.TodoUpdatedData) string {
	if len(data.Items) == 0 {
		return ""
	}

	timestamp := timeStyle.Render(formatTime(time.Now()))
	var lines []string
	header := infoStyle.Render("📋 Todo List")
	lines = append(lines, header, "")

	for i, item := range data.Items {
		var statusIcon string
		var statusStyle lipgloss.Style

		switch item.Status {
		case agent.TodoStatusDone:
			statusIcon = "✓"
			statusStyle = successStyle
		case agent.TodoStatusInProgress:
			statusIcon = "◐"
			statusStyle = warningStyle
		default:
			statusIcon = "○"
			statusStyle = subtleStyle
		}

		content := item.Content

		line := fmt.Sprintf("  %s %s %s", statusStyle.Render(statusIcon), subtleStyle.Render(fmt.Sprintf("#%d", i+1)), content)
		lines = append(lines, line)
	}

	return fmt.Sprintf("\n%s %s\n", timestamp, strings.Join(lines, "\n"))
}

func renderReasoning(data *agent.ReasoningData) string {
	if data.Reasoning == "" {
		return ""
	}

	timestamp := timeStyle.Render(formatTime(time.Now()))
	header := reasoningStyle.Render("🤔 Reasoning")
	reasoning := data.Reasoning

	return fmt.Sprintf("\n%s %s\n\n%s\n", timestamp, header, reasoningStyle.Render(reasoning))
}

func renderError(data *agent.ErrorData) string {
	timestamp := timeStyle.Render(formatTime(time.Now()))
	header := errorStyle.Render("✗ Error")
	divider := subtleStyle.Render(strings.Repeat("─", 40))

	return fmt.Sprintf("\n%s %s\n%s\n\n%s\n", timestamp, header, divider, errorStyle.Render(data.Message))
}

// formatTime formats a time.Time to a string for display
func formatTime(t time.Time) string {
	return t.Format("15:04:05")
}

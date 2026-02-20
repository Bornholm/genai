package chat

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/gdamore/tcell/v2/encoding"
	"github.com/mattn/go-runewidth"
	"github.com/pkg/errors"
)

// UI represents the terminal chat UI
type UI struct {
	session *ChatSession
	screen  tcell.Screen

	// UI state
	mu           sync.RWMutex
	inputBuffer  []rune
	messages     []MessageLine
	scrollOffset int
	inputCursor  int

	// Input history
	inputHistory     []string
	historyIndex     int
	historyTemporary string // Stores current input when navigating history

	// Dimensions
	width  int
	height int

	// State flags
	isStreaming     bool
	streamingBuffer strings.Builder
	toolCallInfo    *ToolCallInfo
}

// MessageLine represents a line in the chat display
type MessageLine struct {
	Role      string // "user", "assistant", "system", "tool"
	Content   string
	IsError   bool
	Style     LineStyle
	Timestamp time.Time
	IsHeader  bool // Marks this as a header line with time/role
}

// LineStyle represents styling for a line
type LineStyle struct {
	Bold      bool
	Italic    bool
	Dim       bool
	Underline bool
}

// ToolCallInfo holds information about an ongoing tool call
type ToolCallInfo struct {
	Name   string
	Params string
	Result string
}

// NewUI creates a new terminal UI
func NewUI(session *ChatSession) (*UI, error) {
	encoding.Register()

	screen, err := tcell.NewScreen()
	if err != nil {
		return nil, errors.Wrap(err, "failed to create screen")
	}

	if err := screen.Init(); err != nil {
		return nil, errors.Wrap(err, "failed to initialize screen")
	}

	ui := &UI{
		session:      session,
		screen:       screen,
		inputBuffer:  make([]rune, 0),
		messages:     make([]MessageLine, 0),
		scrollOffset: 0,
		inputCursor:  0,
		inputHistory: make([]string, 0),
		historyIndex: -1,
	}

	// Set up callbacks
	session.onStreamChunk = ui.onStreamChunk
	session.onToolCall = ui.onToolCall
	session.onToolResult = ui.onToolResult
	session.onComplete = ui.onComplete
	session.onError = ui.onError

	return ui, nil
}

// Run starts the UI event loop
func (ui *UI) Run(ctx context.Context) error {
	defer ui.screen.Fini()

	ui.screen.SetStyle(tcell.StyleDefault.
		Foreground(tcell.ColorWhite).
		Background(tcell.ColorBlack))

	ui.width, ui.height = ui.screen.Size()

	// Create welcome message with provider/model info
	provider, model := ui.session.GetProviderModel()
	welcomeMsg := "Welcome to genai chat!"
	if provider != "" && model != "" {
		welcomeMsg = "Welcome to genai chat! \nProvider: " + provider + " | Model: " + model
	} else if model != "" {
		welcomeMsg = "Welcome to genai chat! Model: " + model
	}
	welcomeMsg += "\nPress Ctrl+C to exit, Ctrl+L to clear."
	ui.addSystemMessage(welcomeMsg)

	// Draw initial state
	ui.draw()

	// Main event loop
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		ev := ui.screen.PollEvent()
		if ev == nil {
			continue
		}

		switch ev := ev.(type) {
		case *tcell.EventKey:
			if err := ui.handleKey(ev); err != nil {
				if errors.Is(err, ErrExit) {
					return nil
				}
				return errors.WithStack(err)
			}
		case *tcell.EventResize:
			ui.width, ui.height = ev.Size()
			ui.draw()
		case *tcell.EventError:
			return errors.New(ev.Error())
		}

		ui.draw()
	}
}

// ErrExit is returned when the user requests to exit
var ErrExit = errors.New("exit requested")

// handleKey processes keyboard input
func (ui *UI) handleKey(ev *tcell.EventKey) error {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	// Handle streaming state - only allow Ctrl+C during streaming
	if ui.isStreaming {
		if ev.Key() == tcell.KeyCtrlC {
			return ErrExit
		}
		return nil
	}

	switch ev.Key() {
	case tcell.KeyCtrlC:
		return ErrExit
	case tcell.KeyCtrlL:
		ui.messages = make([]MessageLine, 0)
		ui.session.Clear()
		ui.scrollOffset = 0
		ui.addSystemMessage("Chat cleared.")
	case tcell.KeyEnter:
		return ui.sendMessage()
	case tcell.KeyBackspace, tcell.KeyBackspace2:
		if ui.inputCursor > 0 && len(ui.inputBuffer) > 0 {
			ui.inputBuffer = append(ui.inputBuffer[:ui.inputCursor-1], ui.inputBuffer[ui.inputCursor:]...)
			ui.inputCursor--
		}
	case tcell.KeyDelete:
		if ui.inputCursor < len(ui.inputBuffer) {
			ui.inputBuffer = append(ui.inputBuffer[:ui.inputCursor], ui.inputBuffer[ui.inputCursor+1:]...)
		}
	case tcell.KeyLeft:
		if ui.inputCursor > 0 {
			ui.inputCursor--
		}
	case tcell.KeyRight:
		if ui.inputCursor < len(ui.inputBuffer) {
			ui.inputCursor++
		}
	case tcell.KeyHome:
		ui.inputCursor = 0
	case tcell.KeyEnd:
		ui.inputCursor = len(ui.inputBuffer)
	case tcell.KeyUp:
		// Navigate input history (if not at first line)
		if len(ui.inputBuffer) == 0 && len(ui.inputHistory) > 0 {
			ui.navigateHistoryUp()
		} else {
			// Scroll up
			if ui.scrollOffset > 0 {
				ui.scrollOffset--
			}
		}
	case tcell.KeyDown:
		// Navigate input history (if in history mode)
		if ui.historyIndex >= 0 {
			ui.navigateHistoryDown()
		} else {
			// Scroll down
			maxScroll := ui.getMaxScroll()
			if ui.scrollOffset < maxScroll {
				ui.scrollOffset++
			}
		}
	case tcell.KeyPgUp:
		// Page up
		ui.scrollOffset -= 5
		if ui.scrollOffset < 0 {
			ui.scrollOffset = 0
		}
	case tcell.KeyPgDn:
		// Page down
		maxScroll := ui.getMaxScroll()
		ui.scrollOffset += 5
		if ui.scrollOffset > maxScroll {
			ui.scrollOffset = maxScroll
		}
	default:
		if ev.Rune() != 0 {
			ui.inputBuffer = append(ui.inputBuffer[:ui.inputCursor], append([]rune{ev.Rune()}, ui.inputBuffer[ui.inputCursor:]...)...)
			ui.inputCursor++
		}
	}

	return nil
}

// sendMessage sends the current input
func (ui *UI) sendMessage() error {
	if len(ui.inputBuffer) == 0 {
		return nil
	}

	message := string(ui.inputBuffer)

	// Add to input history (avoid duplicates)
	if len(ui.inputHistory) == 0 || ui.inputHistory[len(ui.inputHistory)-1] != message {
		ui.inputHistory = append(ui.inputHistory, message)
	}

	// Reset history navigation
	ui.historyIndex = -1
	ui.historyTemporary = ""

	ui.inputBuffer = make([]rune, 0)
	ui.inputCursor = 0

	// Add user message to display
	ui.addUserMessage(message)

	// Set streaming state
	ui.isStreaming = true
	ui.streamingBuffer.Reset()
	ui.toolCallInfo = nil

	// Send message asynchronously
	go func() {
		defer func() {
			ui.mu.Lock()
			ui.isStreaming = false
			ui.mu.Unlock()
			ui.screen.PostEvent(&tcell.EventInterrupt{})
		}()

		if err := ui.session.SendMessage(context.Background(), message); err != nil {
			ui.mu.Lock()
			ui.addErrorMessage(err.Error())
			ui.mu.Unlock()
		}
	}()

	return nil
}

// navigateHistoryUp navigates up in input history
func (ui *UI) navigateHistoryUp() {
	if len(ui.inputHistory) == 0 {
		return
	}

	// Save current input if starting navigation
	if ui.historyIndex == -1 {
		ui.historyTemporary = string(ui.inputBuffer)
	}

	ui.historyIndex++
	if ui.historyIndex >= len(ui.inputHistory) {
		ui.historyIndex = len(ui.inputHistory) - 1
	}

	// Load history item (from end, most recent first)
	idx := len(ui.inputHistory) - 1 - ui.historyIndex
	ui.inputBuffer = []rune(ui.inputHistory[idx])
	ui.inputCursor = len(ui.inputBuffer)
}

// navigateHistoryDown navigates down in input history
func (ui *UI) navigateHistoryDown() {
	if ui.historyIndex == -1 {
		return
	}

	ui.historyIndex--

	if ui.historyIndex < 0 {
		// Restore temporary input
		ui.inputBuffer = []rune(ui.historyTemporary)
		ui.inputCursor = len(ui.inputBuffer)
		ui.historyIndex = -1
	} else {
		// Load history item
		idx := len(ui.inputHistory) - 1 - ui.historyIndex
		ui.inputBuffer = []rune(ui.inputHistory[idx])
		ui.inputCursor = len(ui.inputBuffer)
	}
}

// onStreamChunk handles streaming chunks
func (ui *UI) onStreamChunk(chunk string) {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	ui.streamingBuffer.WriteString(chunk)
	ui.draw()
}

// onToolCall handles tool call events
func (ui *UI) onToolCall(name string, params map[string]any) {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	ui.toolCallInfo = &ToolCallInfo{
		Name:   name,
		Params: paramsToString(params),
	}
	ui.draw()
}

// onToolResult handles tool result events
func (ui *UI) onToolResult(name string, result string) {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	if ui.toolCallInfo != nil {
		ui.toolCallInfo.Result = result
	}
	ui.draw()
}

// onComplete handles completion events
func (ui *UI) onComplete() {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	// Add the complete assistant message
	content := ui.streamingBuffer.String()
	if content != "" {
		ui.addAssistantMessage(content)
	}

	// Add tool call info if present
	if ui.toolCallInfo != nil {
		ui.addToolMessage(ui.toolCallInfo.Name, ui.toolCallInfo.Result)
	}

	ui.streamingBuffer.Reset()
	ui.toolCallInfo = nil
	ui.draw()
}

// onError handles error events
func (ui *UI) onError(err error) {
	ui.mu.Lock()
	defer ui.mu.Unlock()
	ui.addErrorMessage(err.Error())
	ui.draw()
}

// draw renders the UI
func (ui *UI) draw() {
	ui.screen.Clear()

	// Draw messages area
	ui.drawMessages()

	// Draw input area
	ui.drawInput()

	// Draw status bar
	ui.drawStatusBar()

	ui.screen.Show()
}

// drawMessages renders the message history
func (ui *UI) drawMessages() {
	// Calculate available height for messages (reserve 3 lines for input + 1 for status)
	messageHeight := ui.height - 4
	if messageHeight < 1 {
		return
	}

	// Build lines from messages
	lines := ui.buildMessageLines()

	// Calculate scroll position
	maxScroll := len(lines) - messageHeight
	if maxScroll < 0 {
		maxScroll = 0
	}

	// Auto-scroll to bottom if we're at the bottom
	if ui.scrollOffset >= maxScroll || ui.isStreaming {
		ui.scrollOffset = maxScroll
	}

	// Draw visible lines
	for y := 0; y < messageHeight; y++ {
		lineIndex := y + ui.scrollOffset
		if lineIndex >= len(lines) {
			break
		}

		line := lines[lineIndex]

		// Handle empty lines (spacing)
		if line.Content == "" && line.Role == "" {
			continue
		}

		// Handle header lines with special styling
		if line.IsHeader {
			headerStyle := ui.getStyleForRole(line.Role, line.IsError).Bold(true).Dim(true)
			ui.drawFormattedLine(y, line.Content, headerStyle)
			continue
		}

		baseStyle := ui.getStyleForRole(line.Role, line.IsError)

		// Apply line style
		if line.Style.Bold {
			baseStyle = baseStyle.Bold(true)
		}
		if line.Style.Italic {
			baseStyle = baseStyle.Italic(true)
		}
		if line.Style.Dim {
			baseStyle = baseStyle.Dim(true)
		}
		if line.Style.Underline {
			baseStyle = baseStyle.Underline(true)
		}

		// Draw the line with markdown formatting
		ui.drawFormattedLine(y, line.Content, baseStyle)
	}
}

// buildMessageLines converts messages to displayable lines
func (ui *UI) buildMessageLines() []MessageLine {
	lines := make([]MessageLine, 0)

	var lastRole string

	// Add existing messages
	for _, msg := range ui.messages {
		// Add spacing between different message entries
		if lastRole != "" && lastRole != msg.Role {
			lines = append(lines, MessageLine{
				Role:    "",
				Content: "",
			})
		}

		// Add header line with time and role
		if msg.Role != "" {
			timeStr := msg.Timestamp.Format("15:04")
			headerText := ui.getHeaderForRole(msg.Role, timeStr)
			lines = append(lines, MessageLine{
				Role:      msg.Role,
				Content:   headerText,
				IsHeader:  true,
				Timestamp: msg.Timestamp,
			})
		}

		// Add wrapped message content
		wrappedLines := ui.wrapMessage(msg)
		lines = append(lines, wrappedLines...)

		lastRole = msg.Role
	}

	// Add streaming content if active
	if ui.isStreaming && ui.streamingBuffer.Len() > 0 {
		// Add spacing if previous message was different
		if lastRole != "" && lastRole != "assistant" {
			lines = append(lines, MessageLine{
				Role:    "",
				Content: "",
			})
		}

		// Add header for streaming
		timeStr := time.Now().Format("15:04")
		headerText := ui.getHeaderForRole("assistant", timeStr)
		lines = append(lines, MessageLine{
			Role:     "assistant",
			Content:  headerText,
			IsHeader: true,
		})

		streamLines := ui.wrapMessage(MessageLine{
			Role:    "assistant",
			Content: ui.streamingBuffer.String(),
		})
		lines = append(lines, streamLines...)
	}

	// Add tool call info if present
	if ui.toolCallInfo != nil {
		lines = append(lines, MessageLine{
			Role:    "tool",
			Content: "[Tool: " + ui.toolCallInfo.Name + "]",
		})
		if ui.toolCallInfo.Result != "" {
			lines = append(lines, MessageLine{
				Role:    "tool",
				Content: "  Result: " + truncate(ui.toolCallInfo.Result, 100),
			})
		}
	}

	return lines
}

// drawFormattedLine draws a line with basic markdown formatting
func (ui *UI) drawFormattedLine(y int, content string, baseStyle tcell.Style) {
	xPos := 0
	runes := []rune(content)
	i := 0

	for i < len(runes) {
		// Check for markdown patterns
		if runes[i] == '*' || runes[i] == '_' {
			// Look for bold (** or __)
			if i+1 < len(runes) && (runes[i+1] == '*' || runes[i+1] == '_') {
				// Find closing **
				closeIdx := findClosing(runes, i+2, runes[i], runes[i+1])
				if closeIdx != -1 {
					// Draw bold text
					style := baseStyle.Bold(true)
					for j := i + 2; j < closeIdx; j++ {
						if xPos >= ui.width {
							break
						}
						ui.screen.SetContent(xPos, y, runes[j], nil, style)
						xPos += runewidth.RuneWidth(runes[j])
					}
					i = closeIdx + 2
					continue
				}
			}
			// Look for italic (* or _)
			closeIdx := findClosingSingle(runes, i+1, runes[i])
			if closeIdx != -1 {
				// Draw italic text
				style := baseStyle.Italic(true)
				for j := i + 1; j < closeIdx; j++ {
					if xPos >= ui.width {
						break
					}
					ui.screen.SetContent(xPos, y, runes[j], nil, style)
					xPos += runewidth.RuneWidth(runes[j])
				}
				i = closeIdx + 1
				continue
			}
		}

		// Check for inline code (`)
		if runes[i] == '`' {
			closeIdx := findClosingSingle(runes, i+1, '`')
			if closeIdx != -1 {
				// Draw code with different style
				style := baseStyle.Foreground(tcell.ColorDarkCyan)
				for j := i + 1; j < closeIdx; j++ {
					if xPos >= ui.width {
						break
					}
					ui.screen.SetContent(xPos, y, runes[j], nil, style)
					xPos += runewidth.RuneWidth(runes[j])
				}
				i = closeIdx + 1
				continue
			}
		}

		// Regular character
		if xPos >= ui.width {
			break
		}
		ui.screen.SetContent(xPos, y, runes[i], nil, baseStyle)
		xPos += runewidth.RuneWidth(runes[i])
		i++
	}
}

// findClosing finds the closing pair of markdown syntax (e.g., ** or __)
func findClosing(runes []rune, start int, ch1, ch2 rune) int {
	for i := start; i < len(runes)-1; i++ {
		if runes[i] == ch1 && runes[i+1] == ch2 {
			return i
		}
	}
	return -1
}

// findClosingSingle finds the closing single character (e.g., * or _ or `)
func findClosingSingle(runes []rune, start int, ch rune) int {
	for i := start; i < len(runes); i++ {
		if runes[i] == ch {
			return i
		}
	}
	return -1
}

// wrapMessage wraps a message to fit the screen width using proper rune width
// Handles line breaks and basic markdown
func (ui *UI) wrapMessage(msg MessageLine) []MessageLine {
	lines := make([]MessageLine, 0)

	// Split by newlines first to preserve paragraph structure
	paragraphs := strings.Split(msg.Content, "\n")

	linePrefix := "" // No prefix, we use headers now
	firstLine := true

	for paraIdx, paragraph := range paragraphs {
		if paraIdx > 0 {
			// Add empty line between paragraphs
			if firstLine {
				firstLine = false
			}
		}

		// Check for headers
		trimmed := strings.TrimLeft(paragraph, " ")
		indent := len(paragraph) - len(trimmed)
		headerLevel := 0
		for _, ch := range trimmed {
			if ch == '#' {
				headerLevel++
			} else {
				break
			}
		}

		// Process header
		if headerLevel > 0 && headerLevel <= 3 {
			headerText := strings.TrimLeft(trimmed[headerLevel:], " ")
			style := LineStyle{Bold: true}
			if headerLevel == 1 {
				style.Underline = true
			}

			// Wrap header text
			wrappedLines := ui.wrapText(headerText, linePrefix, ui.width)
			for i, wrappedLine := range wrappedLines {
				lineStyle := style
				if i > 0 {
					lineStyle.Underline = false
				}
				lines = append(lines, MessageLine{
					Role:    msg.Role,
					Content: wrappedLine,
					IsError: msg.IsError,
					Style:   lineStyle,
				})
			}

			if firstLine {
				firstLine = false
			}
			continue
		}

		// Check for list items
		listPrefix := ""
		if len(trimmed) > 0 {
			if trimmed[0] == '-' || trimmed[0] == '*' {
				if len(trimmed) > 1 && trimmed[1] == ' ' {
					listPrefix = "• "
					paragraph = strings.Repeat(" ", indent) + listPrefix + trimmed[2:]
				}
			} else if len(trimmed) > 1 && trimmed[0] >= '0' && trimmed[0] <= '9' && trimmed[1] == '.' {
				if len(trimmed) > 2 && trimmed[2] == ' ' {
					listPrefix = string(trimmed[0]) + ". "
					paragraph = strings.Repeat(" ", indent) + listPrefix + trimmed[3:]
				}
			}
		}

		// Wrap regular text
		wrappedLines := ui.wrapText(paragraph, linePrefix, ui.width)
		for _, wrappedLine := range wrappedLines {
			lines = append(lines, MessageLine{
				Role:    msg.Role,
				Content: wrappedLine,
				IsError: msg.IsError,
				Style:   msg.Style,
			})
		}

		if firstLine && len(wrappedLines) > 0 {
			firstLine = false
		}
	}

	return lines
}

// wrapText wraps text to fit within the given width
func (ui *UI) wrapText(text, prefix string, maxWidth int) []string {
	lines := make([]string, 0)

	// Strip markdown formatting for width calculation
	plainText := stripMarkdown(text)
	words := strings.Fields(plainText)
	if len(words) == 0 {
		return lines
	}

	var currentLine strings.Builder
	currentWidth := runewidth.StringWidth(prefix)

	for _, word := range words {
		wordWidth := runewidth.StringWidth(word)
		// Check if adding this word would exceed the width
		if currentWidth+wordWidth+1 > maxWidth && currentLine.Len() > 0 {
			lines = append(lines, prefix+currentLine.String())
			currentLine.Reset()
			currentWidth = 0
			prefix = ""
		}
		if currentLine.Len() > 0 {
			currentLine.WriteString(" ")
			currentWidth++
		}
		currentLine.WriteString(word)
		currentWidth += wordWidth
	}

	if currentLine.Len() > 0 {
		lines = append(lines, prefix+currentLine.String())
	}

	return lines
}

// stripMarkdown removes markdown formatting for width calculation
func stripMarkdown(s string) string {
	var result strings.Builder
	runes := []rune(s)
	i := 0

	for i < len(runes) {
		// Skip bold markers
		if runes[i] == '*' || runes[i] == '_' {
			if i+1 < len(runes) && runes[i+1] == runes[i] {
				// Bold ** or __
				closeIdx := findClosing(runes, i+2, runes[i], runes[i+1])
				if closeIdx != -1 {
					// Write content without markers
					result.WriteString(string(runes[i+2 : closeIdx]))
					i = closeIdx + 2
					continue
				}
			}
			// Italic * or _
			closeIdx := findClosingSingle(runes, i+1, runes[i])
			if closeIdx != -1 {
				result.WriteString(string(runes[i+1 : closeIdx]))
				i = closeIdx + 1
				continue
			}
		}

		// Skip code markers
		if runes[i] == '`' {
			closeIdx := findClosingSingle(runes, i+1, '`')
			if closeIdx != -1 {
				result.WriteString(string(runes[i+1 : closeIdx]))
				i = closeIdx + 1
				continue
			}
		}

		result.WriteRune(runes[i])
		i++
	}

	return result.String()
}

// drawInput renders the input area
func (ui *UI) drawInput() {
	inputY := ui.height - 3

	// Draw input label
	label := "> "
	for x, ch := range label {
		ui.screen.SetContent(x, inputY, ch, nil, tcell.StyleDefault.Bold(true))
	}

	// Draw input buffer with proper rune width
	inputStart := runewidth.StringWidth(label)
	xPos := inputStart
	for _, ch := range ui.inputBuffer {
		if xPos >= ui.width {
			break
		}
		ui.screen.SetContent(xPos, inputY, ch, nil, tcell.StyleDefault)
		xPos += runewidth.RuneWidth(ch)
	}

	// Draw cursor - calculate cursor position based on rune widths
	cursorX := inputStart
	for i := 0; i < ui.inputCursor && i < len(ui.inputBuffer); i++ {
		cursorX += runewidth.RuneWidth(ui.inputBuffer[i])
	}
	if cursorX < ui.width {
		ui.screen.ShowCursor(cursorX, inputY)
	}

	// Draw input border
	borderY := ui.height - 2
	for x := 0; x < ui.width; x++ {
		ui.screen.SetContent(x, borderY, '─', nil, tcell.StyleDefault.Dim(true))
	}
}

// drawStatusBar renders the status bar
func (ui *UI) drawStatusBar() {
	statusY := ui.height - 1

	var status string
	if ui.isStreaming {
		status = "⏳ Generating..."
	} else {
		status = "Ctrl+C: Exit | Ctrl+L: Clear | ↑↓: Scroll"
	}

	// Draw status with proper rune width
	style := tcell.StyleDefault.Reverse(true)
	xPos := 0
	for _, ch := range status {
		if xPos >= ui.width {
			break
		}
		ui.screen.SetContent(xPos, statusY, ch, nil, style)
		xPos += runewidth.RuneWidth(ch)
	}

	// Fill rest of status bar
	for x := xPos; x < ui.width; x++ {
		ui.screen.SetContent(x, statusY, ' ', nil, style)
	}
}

// Helper functions

func (ui *UI) addUserMessage(content string) {
	ui.messages = append(ui.messages, MessageLine{
		Role:      "user",
		Content:   content,
		Timestamp: time.Now(),
	})
}

func (ui *UI) addAssistantMessage(content string) {
	ui.messages = append(ui.messages, MessageLine{
		Role:      "assistant",
		Content:   content,
		Timestamp: time.Now(),
	})
}

func (ui *UI) addSystemMessage(content string) {
	ui.messages = append(ui.messages, MessageLine{
		Role:      "system",
		Content:   content,
		Timestamp: time.Now(),
	})
}

func (ui *UI) addErrorMessage(content string) {
	ui.messages = append(ui.messages, MessageLine{
		Role:      "system",
		Content:   "Error: " + content,
		IsError:   true,
		Timestamp: time.Now(),
	})
}

func (ui *UI) addToolMessage(name, result string) {
	ui.messages = append(ui.messages, MessageLine{
		Role:      "tool",
		Content:   "[Tool: " + name + "] " + truncate(result, 100),
		Timestamp: time.Now(),
	})
}

// getHeaderForRole returns a formatted header with time and role
func (ui *UI) getHeaderForRole(role, timeStr string) string {
	switch role {
	case "user":
		return "┌─[" + timeStr + "] You ─────────────────────────────────────────"
	case "assistant":
		return "┌─[" + timeStr + "] AI ───────────────────────────────────────────"
	case "system":
		return "┌─[" + timeStr + "] System ────────────────────────────────────────"
	case "tool":
		return "┌─[" + timeStr + "] Tool ──────────────────────────────────────────"
	default:
		return ""
	}
}

func (ui *UI) getStyleForRole(role string, isError bool) tcell.Style {
	style := tcell.StyleDefault

	if isError {
		return style.Foreground(tcell.ColorRed)
	}

	switch role {
	case "user":
		return style.Foreground(tcell.ColorGreen)
	case "assistant":
		return style.Foreground(tcell.ColorWhite)
	case "system":
		return style.Foreground(tcell.ColorYellow).Dim(true)
	case "tool":
		return style.Foreground(tcell.ColorDarkCyan)
	default:
		return style
	}
}

func (ui *UI) getPrefixForRole(role string) string {
	// No longer used for display, but kept for compatibility
	return ""
}

func (ui *UI) getMaxScroll() int {
	messageHeight := ui.height - 4
	lines := ui.buildMessageLines()
	maxScroll := len(lines) - messageHeight
	if maxScroll < 0 {
		return 0
	}
	return maxScroll
}

func paramsToString(params map[string]any) string {
	if len(params) == 0 {
		return "{}"
	}
	var sb strings.Builder
	sb.WriteString("{")
	first := true
	for k, v := range params {
		if !first {
			sb.WriteString(", ")
		}
		sb.WriteString(k)
		sb.WriteString(": ")
		sb.WriteString(truncate(formatValue(v), 50))
		first = false
	}
	sb.WriteString("}")
	return sb.String()
}

func formatValue(v any) string {
	switch val := v.(type) {
	case string:
		return val
	case map[string]any:
		return paramsToString(val)
	case []any:
		return "[...]"
	default:
		return truncate(stringify(v), 50)
	}
}

func stringify(v any) string {
	return strings.TrimSpace(strings.ReplaceAll(strings.ReplaceAll(
		strings.ReplaceAll(stringifyValue(v), "\n", " "), "\t", " "), "  ", " "))
}

func stringifyValue(v any) string {
	switch val := v.(type) {
	case string:
		return val
	case map[string]any:
		return paramsToString(val)
	case []any:
		var sb strings.Builder
		sb.WriteString("[")
		for i, item := range val {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(stringifyValue(item))
		}
		sb.WriteString("]")
		return sb.String()
	default:
		return ""
	}
}

func truncate(s string, maxLen int) string {
	if runewidth.StringWidth(s) <= maxLen {
		return s
	}
	// Truncate by rune width
	runes := []rune(s)
	width := 0
	for i, r := range runes {
		rw := runewidth.RuneWidth(r)
		if width+rw > maxLen-3 {
			return string(runes[:i]) + "..."
		}
		width += rw
	}
	return s
}

package proxy

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMapAnthropicModel(t *testing.T) {
	cases := map[string]string{
		"claude-sonnet-5-5":         "anthropic/claude-sonnet-5.5",
		"claude-haiku-5-5-20261001": "anthropic/claude-haiku-5.5",
		"claude-fable-5-2[1m]":      "anthropic/claude-fable-5.2",
		"claude-opus-5.5":           "anthropic/claude-opus-5.5",
		"anthropic/claude-opus-5.5": "anthropic/claude-opus-5.5",
		"openai/gpt-6-luna":         "openai/gpt-6-luna",
		"claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
	}
	for in, want := range cases {
		if got := mapAnthropicModel(in); got != want {
			t.Errorf("mapAnthropicModel(%q) = %q, want %q", in, got, want)
		}
	}
}

func userText(t *testing.T, raw string) (string, []contentFile) {
	t.Helper()
	m := chatMessage{Role: "user", Content: anthropicToOpenAIContent(json.RawMessage(raw))}
	return m.contentParts()
}

func TestAnthropicContentStripsReminders(t *testing.T) {
	text, files := userText(t, `[
		{"type":"text","text":"<system-reminder>\nCLAUDE.md 很长\n</system-reminder>"},
		{"type":"text","text":"你好"},
		{"type":"text","text":"再见"}
	]`)
	if text != "你好\n\n再见" || len(files) != 0 {
		t.Fatalf("got %q files=%d", text, len(files))
	}
}

func TestAnthropicContentOnlyReminderKept(t *testing.T) {
	text, _ := userText(t, `[{"type":"text","text":"<system-reminder>x</system-reminder>"}]`)
	if !strings.Contains(text, "system-reminder") {
		t.Fatalf("全是提醒时应保留原文, got %q", text)
	}
}

func TestAnthropicContentString(t *testing.T) {
	text, _ := userText(t, `"纯文本"`)
	if text != "纯文本" {
		t.Fatalf("got %q", text)
	}
}

func TestAnthropicContentImageAndToolResult(t *testing.T) {
	text, files := userText(t, `[
		{"type":"tool_result","tool_use_id":"x","content":[{"type":"text","text":"ls 输出"}]},
		{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}},
		{"type":"tool_use","id":"y","name":"Bash","input":{}}
	]`)
	if text != "[工具结果]\nls 输出" {
		t.Fatalf("text = %q", text)
	}
	if len(files) != 1 || !strings.HasPrefix(files[0].DataB64, "data:image/png;base64,") || files[0].Filename != "image.png" {
		t.Fatalf("files = %+v", files)
	}
}

func TestEstimateAnthropicInput(t *testing.T) {
	var areq anthropicRequest
	_ = json.Unmarshal([]byte(`{"system":[{"type":"text","text":"abcdefgh"}],"messages":[{"role":"user","content":"abcd"}]}`), &areq)
	if n := estimateAnthropicInput(areq); n != 3 {
		t.Fatalf("got %d", n)
	}
}

// 用户先跑 /model 再提问时，Claude Code 会把命令记录和问题放进同一条消息（2026-11-12 实际请求）。
func TestAnthropicContentStripsLocalCommand(t *testing.T) {
	text, _ := userText(t, `[
		{"type":"text","text":"<local-command-caveat>The command below was run directly in Claude Code, not sent to you as a request.</local-command-caveat>"},
		{"type":"text","text":"<command-name>/model</command-name>\n            <command-message>model</command-message>\n            <command-args></command-args>"},
		{"type":"text","text":"<local-command-stdout>Set model to claude-fable-5-2</local-command-stdout>"},
		{"type":"text","text":"你是谁"}
	]`)
	if text != "你是谁" {
		t.Fatalf("got %q", text)
	}
}

func TestIsClaudeCodeInternalKeepsUserText(t *testing.T) {
	if isClaudeCodeInternal("<command-name> 这个标签是什么意思？") {
		t.Fatal("用户的提问不应被当成内部记录")
	}
}
